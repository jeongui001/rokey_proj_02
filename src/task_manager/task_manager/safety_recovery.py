from std_srvs.srv import Trigger

from handover_interfaces.srv import SetVisionMode

from task_manager.command_parser import Command, Mode, parse_command
from task_manager.task_models import SAFETY_PRIORITY, Safety, State


class SafetyRecovery:
    """Fault 단계 상승과 /robot/recover 절차를 관리하는 mixin."""

    @staticmethod
    def _classify_fault_prefix(text: str) -> str:
        """robot_control과 합의된 /robot/fault 접두어로 안전상태를 분류한다.

        단순히 문자열에 특정 단어가 포함됐는지가 아니라, 정해진 접두어로
        시작하는지를 우선 확인한다. 접두어가 없거나 알 수 없으면 안전하게
        FAULT로 간주한다.
        """
        stripped = (text or '').lstrip()
        if stripped.startswith('PROTECTIVE_STOP:'):
            return Safety.PROTECTIVE_STOP
        if stripped.startswith('EMERGENCY_STOP:'):
            return Safety.EMERGENCY_STOP
        if stripped.startswith('FAULT:'):
            return Safety.FAULT
        return Safety.FAULT

    def _enter_fault(self, detail, safety_state=Safety.FAULT):
        # self.state가 아직 바뀌기 전(이 아래 어떤 코드도 self.state를 직접 바꾸지
        # 않는다 - _request_cancel(lambda: None)의 콜백도 no-op이다)에 재개용
        # 스냅샷을 남긴다 (_capture_resume_snapshot 참고).
        self._capture_resume_snapshot()
        # 새로운 Fault는 복구 진행보다 항상 우선한다 - 진행 중이던 복구 요청(취소
        # 대기, 서비스 응답 대기, 또는 응답 타임아웃 타이머)이 있었다면 모두
        # 무효화한다. generation을 올려두면 이미 보낸 /robot/recover 요청의 지연
        # 응답(성공이든 타임아웃이든)이 나중에 도착해도 무시된다.
        self._recovery_generation += 1
        self._recovery_in_progress = False
        self._stop_recovery_timeout_timer()
        self.safety_state = safety_state
        self._publish_status(detail=detail)
        self.get_logger().error(
            f'task_manager가 {safety_state} 상태로 진입했습니다: {detail}')
        self._request_cancel(lambda: None)

    def _on_fault(self, msg):
        """SAFETY_PRIORITY 기준으로 안전상태를 갱신한다.

        이미 비정상 상태라는 이유만으로 새 Fault를 전부 무시하지 않는다 - 더 높은
        단계(예: PROTECTIVE_STOP -> EMERGENCY_STOP, FAULT -> EMERGENCY_STOP)는
        반드시 즉시 반영한다. 반대로 현재보다 낮은 단계로는 강등하지 않는다
        (EMERGENCY_STOP은 자동으로 낮은 상태가 되지 않는다). 완전히 동일한 메시지가
        같은 등급으로 반복되는 경우에만 중복 처리를 생략한다."""
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

    def _handle_reset(self):
        if self.safety_state == Safety.NORMAL:
            self._publish_status(detail='정상 상태입니다.')
            return
        if self._recovery_in_progress:
            self._publish_status(detail='이미 복구 요청이 진행 중입니다.')
            return
        # 리셋 명령을 받았다고 바로 NORMAL로 바꾸지 않는다. 먼저 RECOVERY_REQUIRED로
        # 전환하고(Vision OFF + 업무 타이머 정리는 _request_cancel이 처리), 진행 중인
        # Action goal이 있다면 취소가 확인된 뒤에만 robot_control의 /robot/recover를
        # 호출한다 (_maybe_start_recovery가 취소 확인 지점에서 이어받는다).
        self.safety_state = Safety.RECOVERY_REQUIRED
        self._recovery_in_progress = True
        self._publish_status(detail='리셋 요청 접수 - 취소 확인 후 복구를 요청합니다.')
        if self._cancel_pending_callback is not None:
            # 이미 다른 사유(Fault 등)로 취소가 진행 중이다 - 그 완료 콜백을 임의로
            # 덮어쓰지 않는다. 그 콜백이 끝나면 _maybe_start_recovery가 이어서
            # 복구 시도 여부를 판단한다.
            return
        self._request_cancel(lambda: None)

    def _maybe_start_recovery(self):
        """진행 중이던 goal 취소가 확인된 직후(또는 취소할 것이 없던 경우 즉시)
        호출된다. 리셋 요청이 걸려 있고, 그 사이 다른 goal이 시작되지 않았고,
        안전상태가 여전히 RECOVERY_REQUIRED일 때만 실제 복구 서비스를 호출한다."""
        if not self._recovery_in_progress:
            return
        if self._goal_in_progress:
            return  # 취소 완료 콜백이 새 goal을 보냈다면(예: place_down) 그것부터 기다린다
        if self.safety_state != Safety.RECOVERY_REQUIRED:
            # 그 사이 새 Fault 등으로 안전상태가 바뀌었다 - 이번 복구 시도는 중단한다.
            self._recovery_in_progress = False
            return
        self._call_recover_service()

    def _call_recover_service(self):
        """goal 취소 완료가 확인된 뒤에만 호출된다. 반드시 비동기로 호출해
        executor를 막지 않는다."""
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
        """현재 살아있는 타이머가 정확히 이 generation 소유일 때만 정리한다.

        오래된(stale) 응답/타임아웃 콜백이 그 사이 시작된 다음 세대의 새 타이머를
        실수로 취소하지 않도록 한다 - generation 일치 여부만으로는 부족하다: 콜백이
        자신의 generation과 self._recovery_generation이 같더라도, 정작 지금 살아있는
        타이머 인스턴스는 자신이 만든 것이 아닐 수 있기 때문에 소유권을 별도로
        확인한다."""
        if self._recovery_timeout_owner_generation == generation:
            self._stop_recovery_timeout_timer()

    def _on_recovery_timeout(self, generation):
        if generation != self._recovery_generation:
            return  # 이미 새 Fault 등으로 세대가 바뀜 - 현재 타이머/상태를 건드리지 않는다
        self._stop_recovery_timeout_timer_if_owned_by(generation)
        if not self._recovery_in_progress:
            return  # 이미 정상 응답(success/실패)으로 처리가 끝남 - 안전하게 무시
        # 응답이 오지 않았다 - generation을 올려 이후 늦게 도착하는 success/실패
        # 응답을 모두 무시하게 하고, RECOVERY_REQUIRED를 유지한 채(자동으로 NORMAL로
        # 전환하지 않고) 사용자가 다시 리셋을 요청할 수 있게 한다.
        self._recovery_generation += 1
        self._recovery_in_progress = False
        self._publish_status(
            detail='/robot/recover 응답 타임아웃 - RECOVERY_REQUIRED 유지, '
                   '다시 리셋을 요청할 수 있습니다.')

    def _on_recover_response(self, future, generation):
        # generation 확인이 가장 먼저다 - 오래된(stale) 응답이면 현재 타이머나
        # _recovery_in_progress를 절대 건드리지 않는다(그 사이 시작된 다음 세대의
        # 새 타이머/상태를 실수로 되돌리지 않기 위함).
        if generation != self._recovery_generation:
            return  # 그 사이 새 Fault(또는 타임아웃)로 세대가 바뀜 - 지연 응답을 무시한다
        self._stop_recovery_timeout_timer_if_owned_by(generation)
        self._recovery_in_progress = False
        if self.safety_state != Safety.RECOVERY_REQUIRED:
            return  # 이미 다른 경로(새 Fault 등)로 안전상태가 바뀜 - 무시한다
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
