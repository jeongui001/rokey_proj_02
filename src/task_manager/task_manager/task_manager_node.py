import json

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from handover_interfaces.action import RobotTask
from handover_interfaces.msg import ToolTrack
from handover_interfaces.srv import SetVisionMode

from task_manager.command_parser import Command, Mode, parse_command


class State:
    IDLE = 'IDLE'
    PARSING = 'PARSING'
    MOVE_TO_WATCH = 'MOVE_TO_WATCH'
    DETECT_TRACK = 'DETECT_TRACK'
    SERVO_PICK = 'SERVO_PICK'
    VERIFY_GRASP = 'VERIFY_GRASP'
    MOVE_SAFE = 'MOVE_SAFE'
    WAIT_PULL = 'WAIT_PULL'
    RELEASE = 'RELEASE'
    HOME = 'HOME'
    MANUAL_MOVE = 'MANUAL_MOVE'
    CANCELLING = 'CANCELLING'


class Safety:
    NORMAL = 'NORMAL'
    PROTECTIVE_STOP = 'PROTECTIVE_STOP'
    FAULT = 'FAULT'
    # 음성 "리셋"만으로는 안전상태를 NORMAL로 되돌리지 않는다.
    # robot_control 쪽에 실제 하드웨어 복구를 확인하는 인터페이스가 아직 없기 때문에,
    # 이 값은 "사용자가 리셋을 요청했다"는 사실만 기록하는 임시 상태이며
    # 실제 로봇 하드웨어에서 안전하다고 간주해서는 안 된다.
    RECOVERY_REQUIRED = 'RECOVERY_REQUIRED'


class TaskManagerNode(Node):
    def __init__(self):
        super().__init__('task_manager')

        self.declare_parameter('detect_track_timeout_s', 5.0)
        self.declare_parameter('verify_grasp_max_retries', 2)
        self.declare_parameter('wait_pull_timeout_s', 60.0)
        self.declare_parameter('cancel_timeout_s', 5.0)

        self.state = State.IDLE
        self.operation_mode = Mode.MANUAL
        self.safety_state = Safety.NORMAL
        self.current_tool = None
        self._verify_grasp_retries = 0
        self._wait_pull_timeout_timer = None
        self._detect_track_timer = None
        self._cancel_timeout_timer = None
        self._goal_in_progress = False
        self._current_goal_handle = None
        self._goal_generation = 0
        self._cancel_pending_callback = None

        self.pub_status = self.create_publisher(String, '/task/status', 10)
        self.sub_command = self.create_subscription(
            String, '/user_command/text', self._on_user_command, 10)
        self.sub_tool_track = self.create_subscription(
            ToolTrack, '/vision/tool_track', self._on_tool_track, 10)
        self.sub_fault = self.create_subscription(
            String, '/robot/fault', self._on_fault, 10)

        self.set_mode_client = self.create_client(SetVisionMode, '/vision/set_mode')
        self.robot_task_client = ActionClient(self, RobotTask, 'robot_task')

    def _safe_call(self, fn, *args, default=None, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.get_logger().warn(f'{fn.__qualname__} not implemented yet: {exc}')
            return default

    def _publish_status(self, detail=''):
        msg = String()
        msg.data = json.dumps({
            'state': self.state,
            'detail': detail,
            'operation_mode': self.operation_mode,
            'safety_state': self.safety_state,
        })
        self.pub_status.publish(msg)

    def _set_state(self, new_state, detail=''):
        self.state = new_state
        self._publish_status(detail)

    # ---- 업무 상태 타이머 정리 (DETECT_TRACK / WAIT_PULL) ----

    def _cancel_detect_track_timer(self):
        if self._detect_track_timer is not None:
            self._detect_track_timer.cancel()
            self._detect_track_timer = None

    def _start_detect_track_timer(self):
        self._cancel_detect_track_timer()
        timeout_s = self.get_parameter('detect_track_timeout_s').value
        self._detect_track_timer = self.create_timer(timeout_s, self._on_detect_track_timeout)

    def _on_detect_track_timeout(self):
        self._cancel_detect_track_timer()
        if self.state != State.DETECT_TRACK:
            return
        self._set_vision_mode(SetVisionMode.Request.OFF)
        self.current_tool = None
        self._set_state(State.IDLE, detail='벨트에 없음 - 감시 시간 초과')

    def _cancel_all_timers(self):
        if self._wait_pull_timeout_timer is not None:
            self._wait_pull_timeout_timer.cancel()
            self._wait_pull_timeout_timer = None
        self._cancel_detect_track_timer()

    # ---- 취소 확인 타임아웃 (cancel_goal_async 호출 후 결과가 오지 않는 경우) ----

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
        self.safety_state = Safety.FAULT
        self._publish_status(detail='취소 확인 타임아웃 - 안전을 위해 FAULT로 전환합니다.')

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
            return
        self._cancel_pending_callback = on_cancelled
        self._start_cancel_timeout_timer()
        if self._current_goal_handle is not None:
            self._current_goal_handle.cancel_goal_async()
        # else: GoalHandle을 아직 받지 못함 - _on_goal_response에서 accept 시 즉시
        # cancel_goal_async()를 호출하거나, reject 시 이 콜백을 안전하게 정리한다.

    # ---- 안전/고장 처리 ----

    @staticmethod
    def _is_protective_stop(text: str) -> bool:
        return 'protective' in text.lower() or '보호' in text

    def _enter_fault(self, detail, safety_state=Safety.FAULT):
        self.safety_state = safety_state
        self._publish_status(detail=detail)
        self._request_cancel(lambda: None)

    def _on_fault(self, msg):
        if self.safety_state != Safety.NORMAL:
            return
        safety_state = (
            Safety.PROTECTIVE_STOP if self._is_protective_stop(msg.data) else Safety.FAULT)
        self._enter_fault(msg.data, safety_state=safety_state)

    # ---- 사용자 명령 처리 (/user_command/text) ----

    def _on_user_command(self, msg):
        command = parse_command(msg.data)
        cmd_type = command['type']

        if cmd_type == Command.RESET:
            self._handle_reset()
            return

        if self.safety_state != Safety.NORMAL:
            self._publish_status(detail='안전 정지 상태 - 리셋이 필요합니다.')
            return

        if cmd_type == Command.MODE_SWITCH:
            self._handle_mode_switch(command['mode'])
        elif cmd_type == Command.STOP:
            self._handle_stop()
        elif cmd_type == Command.MOVE_NAMED:
            self._handle_manual_move(command['named_target'])
        elif cmd_type == Command.FETCH_TOOL:
            self._handle_fetch_tool(command['tool'])
        else:
            self._publish_status(detail='명령을 이해하지 못했습니다. 다시 말씀해주세요.')

    def _handle_reset(self):
        if self.safety_state == Safety.NORMAL:
            self._publish_status(detail='정상 상태입니다.')
            return
        if self.safety_state == Safety.RECOVERY_REQUIRED:
            self._publish_status(detail='이미 리셋 요청됨 - robot_control 복구 확인 대기 중입니다.')
            return
        # robot_control에 실제 하드웨어 복구를 확인할 인터페이스가 없어 NORMAL로 직접
        # 되돌리지 않는다. 사용자의 리셋 의도만 기록하고, 실제 안전 여부는 별도 점검이 필요하다.
        self.safety_state = Safety.RECOVERY_REQUIRED
        self._publish_status(
            detail='리셋 요청 접수 - robot_control 복구 확인 인터페이스 부재로 '
                   '음성 명령만으로는 NORMAL로 복귀하지 않습니다. 수동 점검이 필요합니다.')

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
        if self.operation_mode != Mode.AUTO:
            self._publish_status(detail='공구 전달 명령은 AUTO 모드에서만 지원됩니다.')
            return
        if self.state != State.IDLE:
            return
        self._set_state(State.PARSING, detail=f'tool={tool}')
        self.current_tool = tool
        self._verify_grasp_retries = 0
        self._set_state(State.MOVE_TO_WATCH)
        self._set_vision_mode(SetVisionMode.Request.TRACK_TOOL, self.current_tool)
        self._send_robot_goal('move_named', named_target='watch')

    def _set_vision_mode(self, mode, tool_class=''):
        request = SetVisionMode.Request()
        request.mode = mode
        request.tool_class = tool_class
        self.set_mode_client.call_async(request)

    # ---- RobotTask 액션 goal 송신/수신 ----

    def _send_robot_goal(self, task_type, named_target='', target_pose=None,
                          tool_class='', grasp_width_mm=0.0, grasp_force_n=0.0):
        if self._goal_in_progress:
            self.get_logger().warn(f'{task_type} 요청 무시 - 이미 진행 중인 goal이 있습니다.')
            return
        self._goal_in_progress = True
        self._goal_generation += 1
        generation = self._goal_generation
        goal = RobotTask.Goal()
        goal.task_type = task_type
        goal.named_target = named_target
        if target_pose is not None:
            goal.target_pose = target_pose
        goal.tool_class = tool_class
        goal.grasp_width_mm = grasp_width_mm
        goal.grasp_force_n = grasp_force_n
        future = self.robot_task_client.send_goal_async(
            goal, feedback_callback=self._on_robot_feedback)
        future.add_done_callback(lambda f: self._on_goal_response(f, generation))

    def _on_robot_feedback(self, feedback_msg):
        self._publish_status(detail=f'servo:{feedback_msg.feedback.state}')

    def _on_goal_response(self, future, generation):
        if generation != self._goal_generation:
            return  # 이미 새 goal로 대체된 세대의 응답 - 무시
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._goal_in_progress = False
            self._current_goal_handle = None
            if self._cancel_pending_callback is not None:
                # 취소를 요청했던 goal이 애초에 거절됨 - 취소 완료로 안전하게 정리한다.
                self._stop_cancel_timeout_timer()
                callback = self._cancel_pending_callback
                self._cancel_pending_callback = None
                callback()
                return
            self._enter_fault('goal rejected')
            return
        self._current_goal_handle = goal_handle
        if self._cancel_pending_callback is not None:
            # GoalHandle을 받기 전에 이미 취소가 요청된 상태였다 - 즉시 취소를 건다.
            goal_handle.cancel_goal_async()
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: self._on_robot_result(f, generation))

    def _on_robot_result(self, future, generation):
        if generation != self._goal_generation:
            return  # 취소/대체된 이전 세대 goal의 결과 - 상태에 반영하지 않는다
        self._goal_in_progress = False
        self._current_goal_handle = None
        if self._cancel_pending_callback is not None:
            self._stop_cancel_timeout_timer()
            callback = self._cancel_pending_callback
            self._cancel_pending_callback = None
            callback()
            return
        response = future.result()
        result = response.result
        if self.state == State.MOVE_TO_WATCH:
            self._handle_move_to_watch_result(result)
        elif self.state == State.SERVO_PICK:
            self._handle_servo_pick_result(result)
        elif self.state == State.VERIFY_GRASP:
            self._handle_release_and_retry_result(result)
        elif self.state == State.MOVE_SAFE:
            self._handle_move_safe_result(result)
        elif self.state == State.WAIT_PULL:
            self._handle_wait_pull_result(result)
        elif self.state == State.RELEASE:
            self._handle_release_result(result)
        elif self.state == State.HOME:
            self._handle_home_result(result)
        elif self.state == State.MANUAL_MOVE:
            self._handle_manual_move_result(result)

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
        """시야 내 + approaching이면 True (완화된 트리거 판정, 데모.md 1.3절)."""
        raise NotImplementedError('_check_trigger 구현 필요')

    def _get_grasp_spec(self, tool_class: str):
        """(grasp_width_mm, grasp_force_n) 등록된 공구 스펙을 반환한다."""
        raise NotImplementedError('_get_grasp_spec 구현 필요')

    def _on_tool_track(self, msg):
        if self.state != State.DETECT_TRACK:
            return
        triggered = self._safe_call(self._check_trigger, msg, default=False)
        if not triggered:
            return
        self._cancel_detect_track_timer()
        spec = self._safe_call(self._get_grasp_spec, self.current_tool, default=None)
        width_mm, force_n = spec if spec else (0.0, 0.0)
        self._set_state(State.SERVO_PICK)
        self._send_robot_goal(
            'servo_pick', tool_class=self.current_tool,
            grasp_width_mm=width_mm, grasp_force_n=force_n)

    def _verify_grasp(self, result) -> bool:
        """무게·폭·grip_detected 삼중 확인 (데모.md 2.6/VERIFY_GRASP)."""
        raise NotImplementedError('_verify_grasp 구현 필요')

    def _handle_servo_pick_result(self, result):
        if not result.success:
            if 'torque' in result.message:
                self._enter_fault(result.message)
            else:
                self._set_state(State.DETECT_TRACK, detail=result.message)
                self._start_detect_track_timer()
            return
        self._set_state(State.VERIFY_GRASP)
        verified = self._safe_call(self._verify_grasp, result, default=False)
        if verified:
            self._set_state(State.MOVE_SAFE)
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
            self._set_state(State.WAIT_PULL)
            timeout_s = self.get_parameter('wait_pull_timeout_s').value
            self._wait_pull_timeout_timer = self.create_timer(timeout_s, self._on_wait_pull_timeout)
            self._send_robot_goal('handover_hold')
        else:
            self._enter_fault(result.message)

    def _on_wait_pull_timeout(self):
        self._wait_pull_timeout_timer.cancel()
        self._wait_pull_timeout_timer = None
        if self.state != State.WAIT_PULL:
            return
        # handover_hold goal이 아직 실행 중일 수 있으므로 place_down을 바로 보내지 않고,
        # 취소를 요청한 뒤 결과(취소 확인)를 받은 다음에만 RELEASE로 전이한다.
        self._request_cancel(self._start_release_after_wait_pull_timeout)

    def _start_release_after_wait_pull_timeout(self):
        self._set_state(State.RELEASE, detail='wait_pull timeout - handover_hold 취소 후 place_down')
        self._send_robot_goal('place_down', named_target='place_down')

    def _handle_wait_pull_result(self, result):
        if self._wait_pull_timeout_timer is not None:
            self._wait_pull_timeout_timer.cancel()
            self._wait_pull_timeout_timer = None
        if result.success:
            # RELEASE는 robot_control이 handover_hold 안에서 이미 개방을 완료했음을
            # 표시하기 위한 경유 상태 - 별도 goal 없이 바로 HOME으로 넘어간다.
            self._set_state(State.RELEASE, detail=result.message)
            self._set_state(State.HOME)
            self._set_vision_mode(SetVisionMode.Request.OFF)
            self._send_robot_goal('move_named', named_target='home')
        else:
            self._enter_fault(result.message)

    def _handle_release_result(self, result):
        # WAIT_PULL 타임아웃 후 보낸 place_down goal의 결과 처리
        if result.success:
            self._set_state(State.HOME)
            self._set_vision_mode(SetVisionMode.Request.OFF)
            self._send_robot_goal('move_named', named_target='home')
        else:
            self._enter_fault(result.message)

    def _handle_home_result(self, result):
        if result.success:
            self._set_state(State.IDLE, detail=f'DONE tool={self.current_tool}')
        else:
            self._enter_fault(result.message)


def main(args=None):
    rclpy.init(args=args)
    node = TaskManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
