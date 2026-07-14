from std_srvs.srv import Trigger

from handover_interfaces.srv import SetVisionMode

from task_manager.command_parser import Command, Mode, parse_command
from task_manager.task_models import SAFETY_PRIORITY, Safety, State


class SafetyRecovery:
    """Fault 단계 상승과 /robot/recover 절차를 관리하는 mixin."""

    @staticmethod
    def _classify_fault_prefix(text: str) -> str:
        """robot_control과 합의된 접두어로 분류 - 접두어가 없거나 알 수 없으면 FAULT로 간주(fail-safe)."""
        stripped = (text or '').lstrip()
        if stripped.startswith('PROTECTIVE_STOP:'):
            return Safety.PROTECTIVE_STOP
        if stripped.startswith('EMERGENCY_STOP:'):
            return Safety.EMERGENCY_STOP
        if stripped.startswith('FAULT:'):
            return Safety.FAULT
        return Safety.FAULT

    def _enter_fault(self, detail, safety_state=Safety.FAULT):
        # self.state가 바뀌기 전에 재개용 스냅샷을 남긴다(_capture_resume_snapshot 참고).
        self._capture_resume_snapshot()
        # 새 Fault는 진행 중이던 복구 요청보다 우선한다 - generation을 올려 지연 응답을 무시시킨다.
        self._recovery_generation += 1
        self._recovery_in_progress = False
        self._stop_recovery_timeout_timer()
        self._stop_vision_timeout_timer()
        self.safety_state = safety_state
        self._publish_status(detail=detail)
        self.get_logger().error(
            f'task_manager가 {safety_state} 상태로 진입했습니다: {detail}')
        self._request_cancel(lambda: None)

    def _on_fault(self, msg):
        """SAFETY_PRIORITY 기준으로 갱신 - 더 높은 단계는 즉시 반영하되 낮은 단계로는 강등하지 않는다."""
        new_state = self._classify_fault_prefix(msg.data)
        new_priority = SAFETY_PRIORITY[new_state]
        current_priority = SAFETY_PRIORITY[self.safety_state]
        if new_priority < current_priority:
            return  # 낮은 단계로 강등하지 않는다
        if new_priority == current_priority and msg.data == self._last_fault_detail:
            return  # 완전히 동일한 메시지의 반복 처리는 생략해도 된다
        self._last_fault_detail = msg.data
        self._enter_fault(msg.data, safety_state=new_state)

    # ---- 사용자 명령 처리 (/user_command/text) ----

    def _on_user_command(self, msg):
        command = parse_command(msg.data)
        cmd_type = command['type']

        if cmd_type == Command.RESET:
            self._handle_reset()
            return

        if self.safety_state != Safety.NORMAL:
            self._publish_status(detail='안전 정지 상태 - 리셋이 필요합니다.')
            self.get_logger().warn(
                f'안전 정지 상태라 사용자 명령을 거부했습니다: {msg.data}',
                throttle_duration_sec=1.0)
            return

        if cmd_type == Command.MODE_SWITCH:
            self._handle_mode_switch(command['mode'])
        elif cmd_type == Command.STOP:
            self._handle_stop()
        elif cmd_type == Command.MOVE_NAMED:
            self._handle_manual_move(command['named_target'])
        elif cmd_type == Command.FETCH_TOOL:
            self._handle_fetch_tool(command['tool'])
        elif cmd_type == Command.RESUME:
            self._handle_resume()
        else:
            self._publish_status(detail='명령을 이해하지 못했습니다. 다시 말씀해주세요.')
            self.get_logger().warn(
                f'사용자 명령을 파싱하지 못했습니다: {msg.data}', throttle_duration_sec=1.0)

    def _clear_resume_snapshot(self):
        self._resume_kind = None
        self._resume_state = None
        self._resume_tool = None
        self._resume_grasp_spec = None

    def _handle_reset(self):
        if self.safety_state == Safety.NORMAL:
            if self.state == State.IDLE and self._resume_kind is None:
                self._publish_status(detail='정상 상태입니다.')
                return
            self._clear_resume_snapshot()
            if self.state == State.IDLE:
                self._set_state(State.IDLE, detail='리셋: 대기모드로 복귀')
                return
            if self._cancel_pending_callback is not None:
                self._publish_status(detail='이미 취소 처리 중입니다.')
                return
            self._set_state(State.CANCELLING, detail='리셋 - 작업 취소 중')
            self._request_cancel(lambda: self._set_state(State.IDLE, detail='리셋: 대기모드로 복귀'))
            return
        if self._recovery_in_progress:
            self._publish_status(detail='이미 복구 요청이 진행 중입니다.')
            return
        # 취소 확인 뒤에만 /robot/recover를 호출한다(_maybe_start_recovery가 이어받음).
        self.safety_state = Safety.RECOVERY_REQUIRED
        self._recovery_in_progress = True
        self._publish_status(detail='리셋 요청 접수 - 취소 확인 후 복구를 요청합니다.')
        if self._cancel_pending_callback is not None:
            # 이미 다른 사유로 취소 진행 중 - 콜백을 덮어쓰지 않는다.
            return
        self._request_cancel(lambda: None)

    def _maybe_start_recovery(self):
        """취소 완료 확인 직후 호출 - 리셋 요청이 걸려 있고 안전상태가 여전히 RECOVERY_REQUIRED일 때만 복구를 호출한다."""
        if not self._recovery_in_progress:
            return
        if self._goal_in_progress:
            return  # 취소 완료 콜백이 새 goal을 보냈다면 그것부터 기다린다
        if self.safety_state != Safety.RECOVERY_REQUIRED:
            self._recovery_in_progress = False
            return
        self._call_recover_service()

    def _call_recover_service(self):
        """goal 취소 완료 확인 후에만 호출된다."""
        generation = self._recovery_generation
        if not self.recover_client.service_is_ready():
            self._recovery_in_progress = False
            self._publish_status(
                detail='robot_control /robot/recover 서비스가 아직 준비되지 않았습니다 - '
                       'RECOVERY_REQUIRED 상태를 유지합니다.')
            return
        try:
            future = self.recover_client.call_async(Trigger.Request())
        except Exception as exc:
            self._recovery_in_progress = False
            self._publish_status(
                detail=f'/robot/recover 요청 예외: {exc} - RECOVERY_REQUIRED 유지, '
                       '자동 재시작하지 않습니다.')
            return
        future.add_done_callback(lambda f: self._on_recover_response(f, generation))
        self._start_recovery_timeout_timer(generation)

    # ---- /robot/recover 응답 타임아웃 (one-shot 성격의 재사용 타이머) ----

    def _start_recovery_timeout_timer(self, generation):
        self._stop_recovery_timeout_timer()
        timeout_s = self.get_parameter('recovery_timeout_s').value
        self._recovery_timeout_timer = self.create_timer(
            timeout_s, lambda: self._on_recovery_timeout(generation))
        self._recovery_timeout_owner_generation = generation

    def _stop_recovery_timeout_timer(self):
        if self._recovery_timeout_timer is not None:
            self._recovery_timeout_timer.cancel()
            self._recovery_timeout_timer = None
        self._recovery_timeout_owner_generation = None

    def _stop_recovery_timeout_timer_if_owned_by(self, generation):
        """generation이 같아도 지금 살아있는 타이머가 자신이 만든 게 아닐 수 있어 소유권을 별도 확인한다."""
        if self._recovery_timeout_owner_generation == generation:
            self._stop_recovery_timeout_timer()

    def _on_recovery_timeout(self, generation):
        if generation != self._recovery_generation:
            return
        self._stop_recovery_timeout_timer_if_owned_by(generation)
        if not self._recovery_in_progress:
            return
        # generation 증가로 이후 늦게 도착하는 응답을 모두 무시하고, RECOVERY_REQUIRED를 유지한다.
        self._recovery_generation += 1
        self._recovery_in_progress = False
        self._publish_status(
            detail='/robot/recover 응답 타임아웃 - RECOVERY_REQUIRED 유지, '
                   '다시 리셋을 요청할 수 있습니다.')

    def _on_recover_response(self, future, generation):
        if generation != self._recovery_generation:
            return  # 지연 응답 - 세대가 바뀜
        self._stop_recovery_timeout_timer_if_owned_by(generation)
        self._recovery_in_progress = False
        if self.safety_state != Safety.RECOVERY_REQUIRED:
            return
        try:
            response = future.result()
        except Exception as exc:
            self._publish_status(
                detail=f'/robot/recover 호출 예외: {exc} - RECOVERY_REQUIRED 유지, '
                       '자동 재시작하지 않습니다.')
            return
        if response is None or not response.success:
            message = response.message if response is not None else ''
            self._publish_status(
                detail=f'/robot/recover 실패({message}) - RECOVERY_REQUIRED 유지, '
                       '자동 재시작하지 않습니다.')
            return
        # robot_control이 success=true를 확인해준 경우에만 정상 상태로 되돌린다.
        self.safety_state = Safety.NORMAL
        self.operation_mode = Mode.MANUAL
        self.current_tool = None
        self._active_grasp_spec = None
        self._verify_grasp_retries = 0
        self._cancel_all_timers()
        self._set_vision_mode(SetVisionMode.Request.OFF)
        self._set_state(State.IDLE, detail=f'복구 완료: {response.message}')
