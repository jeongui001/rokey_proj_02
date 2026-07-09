from handover_interfaces.action import RobotTask
from handover_interfaces.srv import SetVisionMode

from task_manager.task_models import Safety, State


# _on_robot_result의 result_state(goal 송신 시점의 self.state) -> (phase, checkpoint_id).
# SERVO_PICK/VERIFY_GRASP/MANUAL_MOVE 결과는 파이프라인 점검.md 체크리스트
# 항목이 아니므로(각각 D/servo_pick_result가 별도로 처리하거나, 수동 이동/재시도라
# 체크리스트 밖) 체크포인트를 발행하지 않는다.
_RESULT_STATE_CHECKPOINTS = {
    State.MOVE_TO_WATCH: ('B', 'move_watch_result_received'),
    State.MOVE_SAFE: ('F', 'handover_safe_result_received'),
    State.APPROACH_HAND: ('H', 'handover_approach_result_received'),
    State.WAIT_PULL: ('I', 'handover_hold_result_received'),
    State.HOME: ('J', 'home_result_received'),
}


class ActionCoordinator:
    """RobotTask Goal 송신·취소·지연 응답을 관리하는 mixin."""

    def _start_cancel_timeout_timer(self):
        self._stop_cancel_timeout_timer()
        timeout_s = self.get_parameter('cancel_timeout_s').value
        self._cancel_timeout_timer = self.create_timer(timeout_s, self._on_cancel_timeout)

    def _stop_cancel_timeout_timer(self):
        if self._cancel_timeout_timer is not None:
            self._cancel_timeout_timer.cancel()
            self._cancel_timeout_timer = None

    def _on_cancel_timeout(self):
        self._stop_cancel_timeout_timer()
        if self._cancel_pending_callback is None:
            return  # 이미 취소가 확인되어 콜백이 실행/정리됨
        # 취소 완료를 확인하지 못했으므로, 나중에 지연 도착하는 result가 있어도
        # 정상 상태로 전이하지 않도록 콜백을 무시(no-op)로 바꿔 남겨둔다.
        self._cancel_pending_callback = lambda: None
        # 취소 타임아웃이면 복구를 시작하지 않는다 - 진행 중이던 복구 요청이 있었다면 중단한다.
        self._recovery_generation += 1
        self._recovery_in_progress = False
        self.safety_state = Safety.FAULT
        self._publish_status(detail='취소 확인 타임아웃 - 안전을 위해 FAULT로 전환합니다.')
        self.get_logger().error(f'goal 취소 완료를 제한 시간 안에 확인하지 못했습니다 (state={self.state}).')

    # ---- Action 취소 (STOP / 모드 전환 / Fault 공용) ----

    def _request_cancel(self, on_cancelled):
        """진행 중인 goal의 취소를 요청하고 Vision을 OFF로 전환한다.

        cancel_goal_async() 호출 자체는 취소 완료를 보장하지 않으므로,
        goal의 최종 result 콜백이 도착한 뒤에만 on_cancelled를 실행한다.
        아직 GoalHandle을 받기 전(accept/reject 응답 대기 중)이면 취소 요청 사실만
        기억해두고, _on_goal_response가 수락/거절에 따라 이어서 처리한다.
        일정 시간 안에 취소가 확인되지 않으면 _on_cancel_timeout이 안전하게 FAULT로 전환한다.
        """
        self._set_vision_mode(SetVisionMode.Request.OFF)
        self._cancel_all_timers()
        if not self._goal_in_progress:
            on_cancelled()
            self._maybe_start_recovery()
            return
        self._cancel_pending_callback = on_cancelled
        self._start_cancel_timeout_timer()
        if self._current_goal_handle is not None:
            try:
                self._current_goal_handle.cancel_goal_async()
            except Exception as exc:
                self._handle_action_future_exception(exc, 'cancel_goal_async')
        # else: GoalHandle을 아직 받지 못함 - _on_goal_response에서 accept 시 즉시
        # cancel_goal_async()를 호출하거나, reject 시 이 콜백을 안전하게 정리한다.

    def _handle_action_future_exception(self, exc, context_label):
        """send_goal_async/cancel_goal_async/future.result() 등 Action 통신 경계에서
        예외가 발생했을 때 공통으로 처리한다.

        pending callback(STOP/모드전환/복구 대기 중이던 콜백)을 성공 처리로 착각해
        호출하지 않는다 - 대신 그 콜백을 버리고 곧바로 FAULT로 전환한다. _goal_generation을
        올려 지연 도착하는 콜백(이미 진행 중이던 send_goal_async/get_result_async의
        결과 등)이 이후에도 상태를 되돌리지 못하게 한다."""
        # self.state가 바뀌기 전에 재개용 스냅샷을 남긴다 (_capture_resume_snapshot 참고,
        # _enter_fault를 거치지 않는 이 예외 경로에서도 동일하게 필요하다).
        self._capture_resume_snapshot()
        self._goal_in_progress = False
        self._current_goal_handle = None
        self._goal_result_state = None
        self._stop_cancel_timeout_timer()
        self._cancel_pending_callback = None
        self._goal_generation += 1
        self._recovery_generation += 1
        self._recovery_in_progress = False
        self._stop_recovery_timeout_timer()
        self.safety_state = Safety.FAULT
        self._set_vision_mode(SetVisionMode.Request.OFF)
        self._publish_status(
            detail=f'{context_label} 예외: {exc} - 안전을 위해 FAULT로 전환합니다.')
        self.get_logger().error(
            f'RobotTask action 통신 경계({context_label})에서 예외가 발생했습니다: {exc} (state={self.state}).')

    # ---- 안전/고장 처리 ----

    def _send_robot_goal(self, task_type, named_target='', tool_class='',
                         grasp_width_mm=0.0, grasp_force_n=0.0):
        if self._goal_in_progress:
            self.get_logger().warn(
                f'{task_type} 요청 무시 - 이미 진행 중인 goal이 있습니다.',
                throttle_duration_sec=1.0)
            return
        self._goal_in_progress = True
        self._goal_generation += 1
        self._goal_result_state = self.state
        generation = self._goal_generation
        goal = RobotTask.Goal()
        goal.task_type = task_type
        goal.named_target = named_target
        goal.tool_class = tool_class
        goal.grasp_width_mm = grasp_width_mm
        goal.grasp_force_n = grasp_force_n
        try:
            future = self.robot_task_client.send_goal_async(
                goal, feedback_callback=self._on_robot_feedback)
        except Exception as exc:
            self._handle_action_future_exception(exc, f'{task_type} send_goal_async')
            return
        future.add_done_callback(lambda f: self._on_goal_response(f, generation))

    def _on_robot_feedback(self, feedback_msg):
        self.get_logger().debug(f'servo:{feedback_msg.feedback.state}')

    def _on_goal_response(self, future, generation):
        if generation != self._goal_generation:
            return  # 이미 새 goal로 대체된 세대의 응답 - 무시
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._handle_action_future_exception(exc, 'goal response')
            return
        if not goal_handle.accepted:
            self._goal_in_progress = False
            self._current_goal_handle = None
            self._goal_result_state = None
            if self._cancel_pending_callback is not None:
                # 취소를 요청했던 goal이 애초에 거절됨 - 취소 완료로 안전하게 정리한다.
                self._stop_cancel_timeout_timer()
                callback = self._cancel_pending_callback
                self._cancel_pending_callback = None
                callback()
                self._maybe_start_recovery()
                return
            self._enter_fault('goal rejected')
            return
        self._current_goal_handle = goal_handle
        if self._cancel_pending_callback is not None:
            # GoalHandle을 받기 전에 이미 취소가 요청된 상태였다 - 즉시 취소를 건다.
            try:
                goal_handle.cancel_goal_async()
            except Exception as exc:
                self._handle_action_future_exception(exc, 'cancel_goal_async')
                return
        try:
            result_future = goal_handle.get_result_async()
        except Exception as exc:
            self._handle_action_future_exception(exc, 'get_result_async')
            return
        result_future.add_done_callback(lambda f: self._on_robot_result(f, generation))

    def _on_robot_result(self, future, generation):
        if generation != self._goal_generation:
            return  # 취소/대체된 이전 세대 goal의 결과 - 상태에 반영하지 않는다
        # Future 예외는 취소 완료 증거가 아니므로 pending callback보다 먼저 확인한다.
        try:
            response = future.result()
        except Exception as exc:
            self._handle_action_future_exception(exc, 'robot result')
            return
        self._goal_in_progress = False
        self._current_goal_handle = None
        result_state = self._goal_result_state or self.state
        self._goal_result_state = None
        if self._cancel_pending_callback is not None:
            self._stop_cancel_timeout_timer()
            callback = self._cancel_pending_callback
            self._cancel_pending_callback = None
            callback()
            self._maybe_start_recovery()
            return
        checkpoint = _RESULT_STATE_CHECKPOINTS.get(result_state)
        if checkpoint is not None:
            phase, checkpoint_id = checkpoint
            self._checkpoint_event(
                phase, checkpoint_id,
                'PASS' if response.result.success else 'FAIL',
                response.result.message or f'{result_state} 결과 수신',
                {'success': bool(response.result.success)})
        handlers = {
            State.MOVE_TO_WATCH: self._handle_move_to_watch_result,
            State.SERVO_PICK: self._handle_servo_pick_result,
            State.VERIFY_GRASP: self._handle_release_and_retry_result,
            State.MOVE_SAFE: self._handle_move_safe_result,
            State.APPROACH_HAND: self._handle_approach_hand_result,
            State.WAIT_PULL: self._handle_wait_pull_result,
            State.HOME: self._handle_home_result,
            State.MANUAL_MOVE: self._handle_manual_move_result,
        }
        handler = handlers.get(result_state)
        if handler is None:
            self._enter_fault(f'결과 처리 상태를 찾을 수 없습니다: {result_state}')
            return
        handler(response.result)
