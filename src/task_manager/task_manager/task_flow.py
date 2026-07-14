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


# CONTINUE_STATES는 그리퍼가 검증된 물체를 쥐고 있어 goal을 그대로 이어가고,
# RETRY_PICK_STATES는 그리퍼 상태가 불확실해 강제로 열고 DETECT_TRACK부터 재시작한다.
_RESUME_CONTINUE_STATES = (State.MOVE_SAFE, State.APPROACH_HAND, State.WAIT_PULL)
_RESUME_RETRY_PICK_STATES = (State.SERVO_PICK, State.VERIFY_GRASP)


class TaskFlow:
    """MANUAL 이동과 AUTO 공구 전달 순서를 관리하는 mixin."""

    def _capture_resume_snapshot(self):
        """FAULT 진입 직전(state 변경 전) 호출 - _enter_fault와 _handle_action_future_exception이 공유."""
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
        """안전상태 NORMAL + state IDLE + 재개 스냅샷 존재 시에만 진행한다(자동 재시작 금지)."""
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
        self._clear_resume_snapshot()

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

        # retry_pick: 그리퍼 상태 불확실 - release_and_retry로 안전하게 열고 재시도
        # (verify_grasp 재시도 횟수에 포함 - _handle_servo_pick_result와 동일 정책).
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
        """취소가 아닌 일시정지 - 재개 가능한 스냅샷을 남기므로 state 변경 전(_set_state 이전)에 캡처한다."""
        if self._cancel_pending_callback is not None:
            # 이미 취소 처리 중 - 기존 콜백을 덮어쓰지 않는다.
            self._publish_status(detail='이미 취소 처리 중입니다.')
            return
        self._capture_resume_snapshot()
        self._set_state(State.CANCELLING, detail='일시정지 - 현재 동작을 멈추는 중')
        self._request_cancel(lambda: self._set_state(
            State.IDLE,
            detail=(
                '일시정지됨 - 재개 버튼으로 이어서 진행할 수 있습니다.'
                if self._resume_kind is not None else '일시정지: 취소 완료')))

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
        # MANUAL 모드도 fetch_tool을 허용한다 - 개별 이동 명령이 추가될 뿐 막지 않는다.
        if not bool(self.get_parameter('auto.config_ready').value):
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
        self._start_vision_timeout_timer(generation)

    # ---- /vision/set_mode 응답 타임아웃 (one-shot 성격의 재사용 타이머) ----

    def _start_vision_timeout_timer(self, generation):
        self._stop_vision_timeout_timer()
        timeout_s = self.get_parameter('vision_mode_timeout_s').value
        self._vision_timeout_timer = self.create_timer(
            timeout_s, lambda: self._on_vision_mode_timeout(generation))
        self._vision_timeout_owner_generation = generation

    def _stop_vision_timeout_timer(self):
        if self._vision_timeout_timer is not None:
            self._vision_timeout_timer.cancel()
            self._vision_timeout_timer = None
        self._vision_timeout_owner_generation = None

    def _stop_vision_timeout_timer_if_owned_by(self, generation):
        """이 generation 소유 타이머일 때만 정리 - 지연 응답이 새 타이머를 취소하지 못하게 한다."""
        if self._vision_timeout_owner_generation == generation:
            self._stop_vision_timeout_timer()

    def _on_vision_mode_timeout(self, generation):
        if generation != self._vision_generation:
            return
        self._stop_vision_timeout_timer_if_owned_by(generation)
        # 세대 증가 - 타임아웃 후 뒤늦은 응답이 로봇 goal을 시작하지 못하게 한다.
        self._vision_generation += 1
        self._enter_fault('/vision/set_mode 응답 타임아웃 - vision_node가 응답하지 않습니다.')

    def _on_vision_mode_response(
            self, future, generation, expected_state, on_success):
        self._stop_vision_timeout_timer_if_owned_by(generation)
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
        """trigger.* 파라미터가 미설정(sentinel -1/빈 문자열)이면 항상 False를 반환한다(fail-closed)."""
        def reject(reason, message, data=None):
            self.get_logger().warn(f'{reason}: {message}', throttle_duration_sec=1.0)
            return False
        if self.current_tool is None:
            return reject('no_current_tool', 'current_tool이 없어 ToolTrack을 무시합니다.')
        if tool_track_msg.tool_class != self.current_tool:
            return reject(
                'class_mismatch', '요청 공구와 검출 공구가 다릅니다.',
                {'expected': self.current_tool, 'actual': tool_track_msg.tool_class})

        confidence = tool_track_msg.confidence
        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            return reject(
                'invalid_confidence',
                'confidence가 NaN/Inf이거나 0.0~1.0 범위를 벗어났습니다.',
                {'confidence': confidence})
        min_confidence = float(self.get_parameter('trigger.min_confidence').value)
        if min_confidence < 0.0 or confidence < min_confidence:
            return reject(
                'low_confidence',
                'confidence가 트리거 기준보다 낮거나 기준값이 미설정입니다.',
                {'confidence': confidence, 'min_confidence': min_confidence})

        if (bool(self.get_parameter('trigger.require_depth_valid').value)
                and not tool_track_msg.depth_valid):
            return reject(
                'depth_invalid', 'depth_valid=false라 트리거하지 않습니다.',
                {'require_depth_valid': True})
        if (bool(self.get_parameter('trigger.require_approaching').value)
                and not tool_track_msg.approaching):
            return reject(
                'not_approaching', 'approaching=false라 트리거하지 않습니다.',
                {'require_approaching': True})

        required_frame_id = self.get_parameter('trigger.required_frame_id').value
        if not required_frame_id or tool_track_msg.header.frame_id != required_frame_id:
            return reject(
                'frame_mismatch',
                'ToolTrack frame_id가 트리거 기준과 다르거나 기준값이 미설정입니다.',
                {'expected': required_frame_id, 'actual': tool_track_msg.header.frame_id})

        max_track_age_s = float(self.get_parameter('trigger.max_track_age_s').value)
        if max_track_age_s < 0.0:
            return reject(
                'track_age_unconfigured', 'max_track_age_s가 미설정이라 트리거하지 않습니다.',
                {'max_track_age_s': max_track_age_s})
        stamp = tool_track_msg.header.stamp
        msg_time_s = stamp.sec + stamp.nanosec * 1e-9
        now_s = self.get_clock().now().nanoseconds * 1e-9
        age_s = now_s - msg_time_s
        if age_s < 0.0 or age_s > max_track_age_s:
            return reject(
                'track_age_out_of_range',
                'ToolTrack 타임스탬프가 너무 오래됐거나 미래 시각입니다.',
                {'age_s': age_s, 'max_track_age_s': max_track_age_s})

        position = tool_track_msg.pose.position
        if not all(math.isfinite(v) for v in (position.x, position.y, position.z)):
            return reject(
                'invalid_position',
                'ToolTrack 좌표가 NaN/Inf입니다.',
                {'x': position.x, 'y': position.y, 'z': position.z})

        self._checkpoint_event(
            'C', 'servo_pick_triggered', 'PASS',
            'servo_pick 트리거 기준을 통과했습니다.',
            {
                'tool_class': tool_track_msg.tool_class,
                'confidence': confidence,
                'frame_id': tool_track_msg.header.frame_id,
                'age_s': age_s,
                'depth_valid': bool(tool_track_msg.depth_valid),
                'approaching': bool(tool_track_msg.approaching),
            },
            throttle_s=1.0)
        return True

    def _get_grasp_spec(self, tool_class: str):
        """값이 미설정(sentinel -1)이거나 min>max 등 모순되면 None - 호출측은 servo_pick을 보내지 않는다."""
        if tool_class not in SUPPORTED_TOOL_CLASSES:
            self._checkpoint_event(
                'C', 'servo_pick_triggered', 'FAIL',
                '지원하지 않는 tool_class라 grasp spec을 읽지 않습니다.',
                {'tool_class': tool_class}, throttle_s=1.0)
            return None
        prefix = f'tools.{tool_class}'
        width_mm = float(self.get_parameter(f'{prefix}.width_mm').value)
        force_n = float(self.get_parameter(f'{prefix}.force_n').value)
        verify_min_width_mm = float(self.get_parameter(f'{prefix}.verify_min_width_mm').value)
        verify_max_width_mm = float(self.get_parameter(f'{prefix}.verify_max_width_mm').value)

        if not all(math.isfinite(v) for v in (
                width_mm, force_n, verify_min_width_mm, verify_max_width_mm)):
            self._checkpoint_event(
                'C', 'servo_pick_triggered', 'FAIL',
                'grasp spec에 NaN/Inf가 포함되어 있습니다.',
                {'tool_class': tool_class}, throttle_s=1.0)
            return None

        if width_mm <= 0.0 or force_n <= 0.0:
            self._checkpoint_event(
                'C', 'servo_pick_triggered', 'FAIL',
                'width_mm 또는 force_n이 미설정/무효입니다.',
                {'tool_class': tool_class, 'width_mm': width_mm, 'force_n': force_n},
                throttle_s=1.0)
            return None
        if verify_min_width_mm < 0.0 or verify_max_width_mm <= 0.0:
            self._checkpoint_event(
                'C', 'servo_pick_triggered', 'FAIL',
                '파지 검증 폭 범위가 미설정/무효입니다.',
                {
                    'tool_class': tool_class,
                    'verify_min_width_mm': verify_min_width_mm,
                    'verify_max_width_mm': verify_max_width_mm,
                },
                throttle_s=1.0)
            return None
        if verify_min_width_mm > verify_max_width_mm:
            self._checkpoint_event(
                'C', 'servo_pick_triggered', 'FAIL',
                'verify_min_width_mm가 verify_max_width_mm보다 큽니다.',
                {
                    'tool_class': tool_class,
                    'verify_min_width_mm': verify_min_width_mm,
                    'verify_max_width_mm': verify_max_width_mm,
                },
                throttle_s=1.0)
            return None

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
            # spec 없음 - 그리퍼를 자동으로 움직이지 않고 IDLE로 복귀한다.
            tool = self.current_tool
            self._active_grasp_spec = None
            self._set_vision_mode(SetVisionMode.Request.OFF)
            self.current_tool = None
            self._set_state(
                State.IDLE,
                detail=f'grasp spec 미설정/유효하지 않음(tool={tool}) - servo_pick을 보내지 않습니다.')
            return
        # 결과 검증 시(_verify_grasp) 동일 spec을 재사용하기 위해 저장한다.
        self._active_grasp_spec = spec
        self._set_state(State.SERVO_PICK)
        self._send_robot_goal(
            'servo_pick', tool_class=self.current_tool,
            grasp_width_mm=spec.width_mm, grasp_force_n=spec.force_n)

    def _verify_grasp(self, result) -> bool:
        """servo_pick 전송 시 spec(self._active_grasp_spec)이 없으면 검증 실패로 간주한다."""
        if not result.success:
            return False
        if not result.grip_detected:
            return False
        if not math.isfinite(result.final_width_mm):
            return False

        spec = self._active_grasp_spec
        if spec is None:
            return False

        if not (spec.verify_min_width_mm <= result.final_width_mm <= spec.verify_max_width_mm):
            return False

        return True

    def _handle_servo_pick_result(self, result):
        if not result.success:
            self._checkpoint_event(
                'D', 'servo_pick_result', 'FAIL', result.message or 'servo_pick 실패',
                {'message': result.message})
            recoverable = ('lost', 'tracking_lost', 'timeout', 'diverged', 'diverging')
            if result.message in recoverable:
                self.get_logger().warn(
                    f'servo_pick 실패({result.message}) - DETECT_TRACK으로 복귀합니다.')
                self._set_state(State.DETECT_TRACK, detail=result.message)
                self._start_detect_track_timer()
            else:
                self.get_logger().error(
                    f'servo_pick 실패 사유({result.message})가 재검출 정책에 없어 FAULT로 전환합니다.')
                self._enter_fault(result.message)
            return
        self._checkpoint_event(
            'D', 'servo_pick_result', 'PASS', 'servo_pick 결과가 정상 반환되었습니다.',
            {'final_width_mm': result.final_width_mm, 'grip_detected': bool(result.grip_detected)})
        self._set_state(State.VERIFY_GRASP)
        verified = self._verify_grasp(result)
        if verified:
            self._checkpoint_event(
                'E', 'grasp_verified', 'PASS', '파지 검증 기준을 통과했습니다.',
                {
                    'grip_detected': bool(result.grip_detected),
                    'final_width_mm': result.final_width_mm,
                })
            self._set_state(State.MOVE_SAFE, detail=f'{self.current_tool} 파지 완료')
            self._send_robot_goal('move_named', named_target='handover_safe')
            return
        self._checkpoint_event(
            'E', 'grasp_verified', 'FAIL', '파지 검증 기준을 통과하지 못했습니다.',
            {
                'grip_detected': bool(result.grip_detected),
                'final_width_mm': result.final_width_mm,
                'active_spec': (
                    self._active_grasp_spec._asdict() if self._active_grasp_spec else None),
            })
        self._verify_grasp_retries += 1
        max_retries = self.get_parameter('verify_grasp_max_retries').value
        if self._verify_grasp_retries > max_retries:
            # 재시도 소진 - 그리퍼를 자동으로 열지 않고 안전 정지로 보고한다.
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
            # vision_node가 아직 hand_track을 채우지 않으면 hardware_ready=false 게이트에 막혀 실패한다.
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
        """handover_hold를 취소하지 않고 계속 들고 대기하며 GUI에 반복 안내만 보낸다.
        TODO: 음성 안내(TTS) 미구현 - 현재는 /task/status.detail 텍스트만."""
        self._wait_pull_timeout_timer.cancel()
        self._wait_pull_timeout_timer = None
        if self.state != State.WAIT_PULL:
            return
        self._publish_status(detail=WAIT_PULL_REMINDER_MESSAGE)
        reminder_interval_s = self.get_parameter('wait_pull_reminder_interval_s').value
        self._wait_pull_timeout_timer = self.create_timer(
            reminder_interval_s, self._on_wait_pull_reminder)

    def _on_wait_pull_reminder(self):
        if self.state != State.WAIT_PULL:
            # 방어적 정리 - 보통 _handle_wait_pull_result가 먼저 취소한다.
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
            # 정상 완주 - 오래된 재개 스냅샷을 정리한다.
            self._clear_resume_snapshot()
            self._set_state(State.IDLE, detail=detail)
        else:
            self._enter_fault(result.message)
