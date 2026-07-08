import math

from handover_interfaces.srv import SetVisionMode

from task_manager.command_parser import Mode
from task_manager.task_models import (
    GraspSpec,
    Safety,
    State,
    SUPPORTED_TOOL_CLASSES,
    WAIT_PULL_REMINDER_MESSAGE,
)


# 재개(resume) 가능 상태 분류. 그리퍼가 이미 검증된 물체를 쥐고 있다는 게 확실한
# 구간(_RESUME_CONTINUE_STATES)은 하던 goal을 그대로 이어서 보낸다. 반대로 지금
# 그리퍼가 정확히 어떤 상태인지 확신할 수 없는 구간(_RESUME_RETRY_PICK_STATES,
# 접근~하강~닫기 도중이거나 방금 닫혀 검증이 안 끝난 상태)은 절대 그대로 이어가지
# 않고, release_and_retry와 동일하게 그리퍼를 강제로 열고 watch로 돌아가
# DETECT_TRACK부터 다시 시작한다(_verify_grasp_retries에 포함되어, 반복되면
# 결국 수동 확인을 요구한다). 나머지 상태(IDLE/MOVE_TO_WATCH/DETECT_TRACK/HOME/
# MANUAL_MOVE/CANCELLING)는 재개 개념 자체가 없다 - 물체를 안 쥐고 있거나
# 사용자가 직접 제어 중이거나 전환 중이라, 그냥 원래 하던 명령을 다시 보내면 된다.
_RESUME_CONTINUE_STATES = (State.MOVE_SAFE, State.APPROACH_HAND, State.WAIT_PULL)
_RESUME_RETRY_PICK_STATES = (State.SERVO_PICK, State.VERIFY_GRASP)


class TaskFlow:
    """MANUAL 이동과 AUTO 공구 전달 순서를 관리하는 mixin."""

    def _capture_resume_snapshot(self):
        """FAULT 진입 직전(아직 self.state가 바뀌기 전) 호출된다 - _enter_fault와
        _handle_action_future_exception 양쪽에서 공유한다."""
        if self.state in _RESUME_CONTINUE_STATES:
            self._resume_kind = 'continue'
            self._resume_state = self.state
            self._resume_tool = self.current_tool
            self._resume_grasp_spec = self._active_grasp_spec
        elif self.state in _RESUME_RETRY_PICK_STATES:
            self._resume_kind = 'retry_pick'
            self._resume_state = self.state
            self._resume_tool = self.current_tool
            self._resume_grasp_spec = None
        else:
            self._resume_kind = None
            self._resume_state = None
            self._resume_tool = None
            self._resume_grasp_spec = None

    def _handle_resume(self):
        """"재개" 명령(/user_command/text) 처리. 안전상태 NORMAL + state IDLE +
        저장된 재개 스냅샷이 있을 때만 진행한다 - 그 외에는 명확한 사유를 GUI에
        보고하고 아무 것도 하지 않는다(자동 재시작 금지)."""
        if self.safety_state != Safety.NORMAL:
            self._publish_status(detail='재개 불가 - 안전상태가 NORMAL이 아닙니다.')
            return
        if self.state != State.IDLE:
            self._publish_status(detail='재개 불가 - 이미 다른 작업이 진행 중입니다.')
            return
        if self._resume_kind is None:
            self._publish_status(detail='재개할 이전 작업이 없습니다.')
            return

        kind = self._resume_kind
        tool = self._resume_tool
        grasp_spec = self._resume_grasp_spec
        resume_state = self._resume_state
        # 재개는 1회성이다 - 실행을 시작하는 시점에 스냅샷을 지워 중복 재개를 막는다.
        self._resume_kind = None
        self._resume_state = None
        self._resume_tool = None
        self._resume_grasp_spec = None

        self.current_tool = tool
        if kind == 'continue':
            self._active_grasp_spec = grasp_spec
            if resume_state == State.MOVE_SAFE:
                self._set_state(State.MOVE_SAFE, detail=f'{tool} 재개 - handover_safe로 이동')
                self._send_robot_goal('move_named', named_target='handover_safe')
            elif resume_state == State.APPROACH_HAND:
                self._set_state(State.APPROACH_HAND, detail=f'{tool} 재개 - 작업자를 찾는 중')
                self._start_after_vision_mode(
                    SetVisionMode.Request.TRACK_HAND, '',
                    State.APPROACH_HAND,
                    lambda: self._send_robot_goal('handover_approach'))
            elif resume_state == State.WAIT_PULL:
                self._set_state(State.WAIT_PULL, detail=f'{tool} 재개 - 당김 대기')
                timeout_s = self.get_parameter('wait_pull_timeout_s').value
                self._wait_pull_timeout_timer = self.create_timer(
                    timeout_s, self._on_wait_pull_timeout)
                self._send_robot_goal('handover_hold')
            return

        # kind == 'retry_pick': SERVO_PICK/VERIFY_GRASP 중단 - 그리퍼 상태가
        # 불확실하므로 release_and_retry와 동일하게 안전하게 열고 watch로 돌아가
        # DETECT_TRACK부터 다시 시작한다. 기존 verify_grasp 재시도 횟수에 포함시켜,
        # 반복되면 결국 수동 확인을 요구한다(_handle_servo_pick_result와 동일 정책).
        self._verify_grasp_retries += 1
        max_retries = self.get_parameter('verify_grasp_max_retries').value
        if self._verify_grasp_retries > max_retries:
            self.current_tool = None
            self._enter_fault(
                '재개 재시도 초과 - 파지 상태가 불확실합니다. 수동 점검이 필요합니다.')
            return
        self._set_state(State.VERIFY_GRASP, detail=f'{tool} 재개 - 안전하게 그리퍼 열고 재시도')
        self._send_robot_goal('release_and_retry')

    def _handle_mode_switch(self, mode):
        if self.state == State.IDLE and not self._goal_in_progress:
            self.operation_mode = mode
            self._set_vision_mode(SetVisionMode.Request.OFF)
            self._publish_status(detail=f'{mode} 모드로 전환')
            return
        if self._cancel_pending_callback is not None:
            # 이미 취소 처리 중 - 기존 콜백을 덮어쓰지 않는다.
            self._publish_status(detail='이미 취소 처리 중입니다.')
            return
        self._set_state(State.CANCELLING, detail=f'{mode} 모드 전환 대기 - 작업 취소 중')
        self._request_cancel(lambda: self._finish_mode_switch(mode))

    def _finish_mode_switch(self, mode):
        self.operation_mode = mode
        self._set_state(State.IDLE, detail=f'{mode} 모드로 전환')

    def _handle_stop(self):
        if self._cancel_pending_callback is not None:
            # 이미 취소 처리 중 - 기존 콜백을 덮어쓰지 않는다.
            self._publish_status(detail='이미 취소 처리 중입니다.')
            return
        self._set_state(State.CANCELLING, detail='정지 명령 - 작업 취소 중')
        self._request_cancel(lambda: self._set_state(State.IDLE, detail='정지: 작업 취소 완료'))

    def _handle_manual_move(self, named_target):
        if self.operation_mode != Mode.MANUAL:
            self._publish_status(detail='이동 명령은 MANUAL 모드에서만 지원됩니다.')
            return
        if self._goal_in_progress:
            self._publish_status(detail='이전 명령 처리 중입니다.')
            return
        self._set_state(State.MANUAL_MOVE, detail=f'move_named:{named_target}')
        self._send_robot_goal('move_named', named_target=named_target)

    def _handle_fetch_tool(self, tool):
        # AUTO/MANUAL 모드 구분 없이 공구 이름이 인식되면(음성/GUI 텍스트 동일 경로)
        # 전체 자동 픽업+전달 시퀀스를 시작한다 - MANUAL은 개별 이동 명령을 추가로
        # 더 허용할 뿐, fetch_tool 자체를 막지 않는다.
        if not bool(self.get_parameter('auto.config_ready').value):
            # AUTO 모드 전환 자체는 허용하되, 실기에서 미합의 trigger/grasp spec
            # 값으로 추측 동작하지 않도록 실제 goal 송신은 막는다.
            self._publish_status(
                detail='AUTO 설정값 미확정(auto.config_ready=false) - '
                       '물체 가져오기 명령을 실행하지 않습니다.')
            return
        if self.state != State.IDLE:
            return
        self.current_tool = tool
        self._active_grasp_spec = None
        self._verify_grasp_retries = 0
        self._set_state(State.MOVE_TO_WATCH, detail=f'{tool} 가져오는 중')
        self._start_after_vision_mode(
            SetVisionMode.Request.TRACK_TOOL,
            self.current_tool,
            State.MOVE_TO_WATCH,
            lambda: self._send_robot_goal('move_named', named_target='watch'))

    def _set_vision_mode(self, mode, tool_class=''):
        self._vision_generation += 1
        if not self.set_mode_client.service_is_ready():
            self.get_logger().warn('/vision/set_mode 서비스가 준비되지 않았습니다.')
            return False
        request = SetVisionMode.Request()
        request.mode = mode
        request.tool_class = tool_class
        try:
            return self.set_mode_client.call_async(request)
        except Exception as exc:
            self.get_logger().error(f'/vision/set_mode 요청 실패: {exc}')
            return False

    def _start_after_vision_mode(
            self, mode, tool_class, expected_state, on_success):
        future = self._set_vision_mode(mode, tool_class)
        generation = self._vision_generation
        if future is None:  # 간단한 mock과의 하위 호환
            on_success()
            return
        if future is False:
            self._enter_fault('/vision/set_mode 서비스를 사용할 수 없습니다.')
            return
        future.add_done_callback(
            lambda done: self._on_vision_mode_response(
                done, generation, expected_state, on_success))

    def _on_vision_mode_response(
            self, future, generation, expected_state, on_success):
        if generation != self._vision_generation or self.state != expected_state:
            return
        try:
            response = future.result()
        except Exception as exc:
            self._enter_fault(f'/vision/set_mode 응답 예외: {exc}')
            return
        if response is None or not response.success:
            message = response.message if response is not None else ''
            self._enter_fault(f'/vision/set_mode 실패: {message}')
            return
        on_success()

    # ---- RobotTask 액션 goal 송신/수신 ----

    def _handle_manual_move_result(self, result):
        if result.success:
            self._set_state(State.IDLE, detail=result.message)
        else:
            self._enter_fault(result.message)

    def _handle_move_to_watch_result(self, result):
        if result.success:
            self._set_state(State.DETECT_TRACK)
            self._start_detect_track_timer()
        else:
            self._enter_fault(result.message)

    def _check_trigger(self, tool_track_msg) -> bool:
        """DETECT_TRACK 중 수신한 ToolTrack이 servo_pick 트리거 조건을 만족하는지
        판정한다. 조건은 모두 ROS 파라미터로 관리하며(trigger.*), 미합의 값은
        sentinel(-1/빈 문자열)로 두어 항상 False(트리거 안 됨)를 반환하게 한다
        (config_ready=false와 별개의 방어선)."""
        if self.current_tool is None:
            return False
        if tool_track_msg.tool_class != self.current_tool:
            return False

        confidence = tool_track_msg.confidence
        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            return False  # NaN/Inf 또는 유효 범위(0.0~1.0) 밖의 confidence는 항상 거부
        min_confidence = float(self.get_parameter('trigger.min_confidence').value)
        if min_confidence < 0.0 or confidence < min_confidence:
            return False

        if (bool(self.get_parameter('trigger.require_depth_valid').value)
                and not tool_track_msg.depth_valid):
            return False
        if (bool(self.get_parameter('trigger.require_approaching').value)
                and not tool_track_msg.approaching):
            return False

        required_frame_id = self.get_parameter('trigger.required_frame_id').value
        if not required_frame_id or tool_track_msg.header.frame_id != required_frame_id:
            return False

        max_track_age_s = float(self.get_parameter('trigger.max_track_age_s').value)
        if max_track_age_s < 0.0:
            return False
        stamp = tool_track_msg.header.stamp
        msg_time_s = stamp.sec + stamp.nanosec * 1e-9
        now_s = self.get_clock().now().nanoseconds * 1e-9
        age_s = now_s - msg_time_s
        if age_s < 0.0 or age_s > max_track_age_s:
            return False

        position = tool_track_msg.pose.position
        if not all(math.isfinite(v) for v in (position.x, position.y, position.z)):
            return False

        return True

    def _get_grasp_spec(self, tool_class: str):
        """등록된 공구별 grasp spec(GraspSpec)을 파라미터(tools.<tool_class>.*)에서
        읽어 반환한다. 값이 미설정(sentinel -1)이거나 앞뒤가 맞지 않으면(min>max 등)
        None을 반환한다 - 호출측은 이 경우 (0.0, 0.0) 같은 값으로 조용히 servo_pick을
        보내지 않아야 한다."""
        if tool_class not in SUPPORTED_TOOL_CLASSES:
            return None
        prefix = f'tools.{tool_class}'
        width_mm = float(self.get_parameter(f'{prefix}.width_mm').value)
        force_n = float(self.get_parameter(f'{prefix}.force_n').value)
        verify_min_width_mm = float(self.get_parameter(f'{prefix}.verify_min_width_mm').value)
        verify_max_width_mm = float(self.get_parameter(f'{prefix}.verify_max_width_mm').value)

        if not all(math.isfinite(v) for v in (
                width_mm, force_n, verify_min_width_mm, verify_max_width_mm)):
            return None  # NaN/Inf가 하나라도 있으면 신뢰할 수 없다

        if width_mm <= 0.0 or force_n <= 0.0:
            return None  # 미설정 - 추측값으로 servo_pick을 보내지 않는다
        if verify_min_width_mm < 0.0 or verify_max_width_mm <= 0.0:
            return None
        if verify_min_width_mm > verify_max_width_mm:
            return None  # 설정 오류

        return GraspSpec(
            width_mm=width_mm, force_n=force_n,
            verify_min_width_mm=verify_min_width_mm,
            verify_max_width_mm=verify_max_width_mm)

    def _on_tool_track(self, msg):
        if self.state != State.DETECT_TRACK:
            return
        if not self._check_trigger(msg):
            return
        self._cancel_detect_track_timer()
        spec = self._get_grasp_spec(self.current_tool)
        if spec is None:
            # grasp spec이 없거나 잘못됨 - RG2/RobotTask goal을 보내지 않고, 명확한
            # IDLE 복귀 정책을 적용한다(그리퍼를 자동으로 움직이지 않는다).
            tool = self.current_tool
            self._active_grasp_spec = None
            self._set_vision_mode(SetVisionMode.Request.OFF)
            self.current_tool = None
            self._set_state(
                State.IDLE,
                detail=f'grasp spec 미설정/유효하지 않음(tool={tool}) - servo_pick을 보내지 않습니다.')
            return
        # 이 servo_pick에 사용한 spec을 저장해 두어, 결과가 왔을 때 같은 설정으로
        # 검증한다 (_verify_grasp 참고).
        self._active_grasp_spec = spec
        self._set_state(State.SERVO_PICK)
        self._send_robot_goal(
            'servo_pick', tool_class=self.current_tool,
            grasp_width_mm=spec.width_mm, grasp_force_n=spec.force_n)

    def _verify_grasp(self, result) -> bool:
        """grip_detected와 final_width_mm로 파지 성공 여부를 검증한다.

        servo_pick을 보낼 때 사용한 grasp spec(self._active_grasp_spec)을 그대로
        재사용한다 - spec이 없으면(예: 재시작 등으로 유실) 검증을 성공으로 간주하지
        않는다."""
        if not result.success:
            return False
        if not result.grip_detected:
            return False
        if not math.isfinite(result.final_width_mm):
            return False

        spec = self._active_grasp_spec
        if spec is None:
            return False  # 검증 기준(spec) 자체가 없다 - 성공으로 간주하지 않는다

        if not (spec.verify_min_width_mm <= result.final_width_mm <= spec.verify_max_width_mm):
            return False

        return True

    def _handle_servo_pick_result(self, result):
        if not result.success:
            if result.message in ('lost', 'timeout', 'diverged'):
                self._set_state(State.DETECT_TRACK, detail=result.message)
                self._start_detect_track_timer()
            else:
                self._enter_fault(result.message)
            return
        self._set_state(State.VERIFY_GRASP)
        verified = self._verify_grasp(result)
        if verified:
            self._set_state(State.MOVE_SAFE, detail=f'{self.current_tool} 파지 완료')
            self._send_robot_goal('move_named', named_target='handover_safe')
            return
        self._verify_grasp_retries += 1
        max_retries = self.get_parameter('verify_grasp_max_retries').value
        if self._verify_grasp_retries > max_retries:
            # 파지 여부가 불확실한 채로 재시도를 모두 소진했다. 물체를 놓치지 않도록
            # 그리퍼를 자동으로 열거나 다른 goal을 보내지 않고, 안전 정지로 보고한다.
            self._enter_fault(
                '파지 검증 실패 - 재시도 초과. 그리퍼를 자동으로 열지 않았습니다. '
                '수동 점검이 필요합니다.')
            return
        self._send_robot_goal('release_and_retry')

    def _handle_release_and_retry_result(self, result):
        if result.success:
            self._set_state(State.DETECT_TRACK)
            self._start_detect_track_timer()
        else:
            self._enter_fault(result.message)

    def _handle_move_safe_result(self, result):
        if result.success:
            # handover_safe 도착 - 작업대가 보이는 자세에서 YOLO로 손을 찾아 접근한다
            # (robot_control의 handover_approach - movel 기반 단발성 이동으로 구현됨,
            # /vision/hand_pose 1회 수신 후 이동. vision_node의 손 추적이 아직
            # hand_pose를 채워 보내지 않으면 handover_approach.timeout_s로 실패한다).
            self._set_state(State.APPROACH_HAND, detail='작업자를 찾는 중')
            self._start_after_vision_mode(
                SetVisionMode.Request.TRACK_HAND,
                '',
                State.APPROACH_HAND,
                lambda: self._send_robot_goal('handover_approach'))
        else:
            self._enter_fault(result.message)

    def _handle_approach_hand_result(self, result):
        if result.success:
            self._set_state(State.WAIT_PULL)
            timeout_s = self.get_parameter('wait_pull_timeout_s').value
            self._wait_pull_timeout_timer = self.create_timer(timeout_s, self._on_wait_pull_timeout)
            self._send_robot_goal('handover_hold')
        else:
            self._enter_fault(result.message)

    def _on_wait_pull_timeout(self):
        """wait_pull_timeout_s가 지나도 사람이 도구를 가져가지 않은 경우.

        handover_hold를 취소하거나 place_down으로 옮기지 않는다 - 로봇은 그
        자리에서 계속 들고 대기하고, 대신 GUI에 반복 안내만 보낸다
        (wait_pull_reminder_interval_s 간격). 실제로 가져가면(pull 확정) 기존
        _handle_wait_pull_result 성공 경로가 그대로 HOME으로 전이시킨다.
        TODO(추후 구현): TTS 등 음성 안내는 아직 만들지 않았다 - 지금은
        /task/status.detail을 통해 GUI에 텍스트로만 표시한다.
        """
        self._wait_pull_timeout_timer.cancel()
        self._wait_pull_timeout_timer = None
        if self.state != State.WAIT_PULL:
            return
        self._publish_status(detail=WAIT_PULL_REMINDER_MESSAGE)
        reminder_interval_s = self.get_parameter('wait_pull_reminder_interval_s').value
        self._wait_pull_timeout_timer = self.create_timer(
            reminder_interval_s, self._on_wait_pull_reminder)

    def _on_wait_pull_reminder(self):
        """_on_wait_pull_timeout 이후 반복 호출되는 안내 타이머(주기적)."""
        if self.state != State.WAIT_PULL:
            # pull이 확정되어 이미 다른 상태로 전이했다면 타이머를 정리한다
            # (일반적으로는 _handle_wait_pull_result가 먼저 취소하지만, 방어적으로
            # 한 번 더 확인한다).
            if self._wait_pull_timeout_timer is not None:
                self._wait_pull_timeout_timer.cancel()
                self._wait_pull_timeout_timer = None
            return
        self._publish_status(detail=WAIT_PULL_REMINDER_MESSAGE)

    def _handle_wait_pull_result(self, result):
        if self._wait_pull_timeout_timer is not None:
            self._wait_pull_timeout_timer.cancel()
            self._wait_pull_timeout_timer = None
        if result.success:
            self._set_state(State.HOME, detail='작업자에게 전달 완료')
            self._set_vision_mode(SetVisionMode.Request.OFF)
            self._send_robot_goal('move_named', named_target='home')
        else:
            self._enter_fault(result.message)

    def _handle_home_result(self, result):
        if result.success:
            detail = f'DONE tool={self.current_tool}'
            self.current_tool = None
            self._active_grasp_spec = None
            # 정상적으로 한 사이클을 완주했다 - 남아있을 수 있는 오래된 재개
            # 스냅샷(예: 이번 사이클 이전의 FAULT 기록)을 정리한다.
            self._resume_kind = None
            self._resume_state = None
            self._resume_tool = None
            self._resume_grasp_spec = None
            self._set_state(State.IDLE, detail=detail)
        else:
            self._enter_fault(result.message)
