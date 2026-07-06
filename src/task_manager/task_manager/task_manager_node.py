import json

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from handover_interfaces.action import RobotTask
from handover_interfaces.msg import ToolTrack
from handover_interfaces.srv import SetVisionMode


# 전체 계획.md 3절의 상태 다이어그램을 그대로 문자열 상수로 옮긴 것.
# IDLE -> PARSING -> MOVE_TO_WATCH -> DETECT_TRACK -> SERVO_PICK -> VERIFY_GRASP
#      -> MOVE_SAFE -> TRACK_HAND -> WAIT_PULL -> RELEASE -> HOME -> IDLE
# 어느 상태에서든 /robot/fault 수신 시 FAULT로 빠진다(수동 리셋 전까지 복귀 없음).
class State:
    IDLE = 'IDLE'
    PARSING = 'PARSING'
    MOVE_TO_WATCH = 'MOVE_TO_WATCH'
    DETECT_TRACK = 'DETECT_TRACK'
    SERVO_PICK = 'SERVO_PICK'
    VERIFY_GRASP = 'VERIFY_GRASP'
    MOVE_SAFE = 'MOVE_SAFE'
    TRACK_HAND = 'TRACK_HAND'
    WAIT_PULL = 'WAIT_PULL'
    RELEASE = 'RELEASE'
    HOME = 'HOME'
    FAULT = 'FAULT'


class TaskManagerNode(Node):
    """명령 해석과 상태머신 감독 주체(전체 계획.md 1.3절).

    좌표 계산은 하지 않는다 - robot_control에 goal(task_type)만 던지고,
    돌아오는 result/feedback을 보고 다음 상태로 넘어가는 감독 역할만 한다.
    """

    def __init__(self):
        super().__init__('task_manager') # 노드명 task_manager

        self.declare_parameter('detect_track_max_cycles', 3)
        self.declare_parameter('verify_grasp_max_retries', 2)
        self.declare_parameter('wait_pull_timeout_s', 60.0)
        self.declare_parameter('hand_detect_timeout_s', 5.0)

        self.state = State.IDLE # 초기 상태 IDLE
        self.current_tool = None
        self._detect_track_cycles = 0      # DETECT_TRACK에서 트리거 미검출 횟수
        self._verify_grasp_retries = 0     # VERIFY_GRASP 실패 후 재시도 횟수
        self._hand_timeout_timer = None    # TRACK_HAND 손 미검출 타임아웃
        self._wait_pull_timeout_timer = None  # WAIT_PULL 당김 미검출 타임아웃

        self.pub_status = self.create_publisher(String, '/task/status', 10)  # 서브스크라이버: handover_ui(rclpy 직접 구독)
        self.sub_command = self.create_subscription(
            String, '/user_command/text', self._on_user_command, 10)  # 퍼블리셔: stt_node, handover_ui(rclpy 직접 구독)
        self.sub_tool_track = self.create_subscription(
            ToolTrack, '/vision/tool_track', self._on_tool_track, 10)  # 퍼블리셔: vision_node
        self.sub_hand_pose = self.create_subscription(
            PoseStamped, '/vision/hand_pose', self._on_hand_pose, 10)  # 퍼블리셔: vision_node
        self.sub_fault = self.create_subscription(
            String, '/robot/fault', self._on_fault, 10)  # 퍼블리셔: robot_control

        self.set_mode_client = self.create_client(SetVisionMode, '/vision/set_mode')  # 서버: vision_node
        self.robot_task_client = ActionClient(self, RobotTask, 'robot_task')  # 서버: robot_control

    def _safe_call(self, fn, *args, default=None, **kwargs):
        """아직 구현 안 된(NotImplementedError) 메서드를 호출해도 노드가 죽지 않게 감싸는 헬퍼.
        이 파일의 스텁 메서드(_call_llm 등)를 부르는 곳은 전부 이걸 통해서 부른다."""
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.get_logger().warn(f'{fn.__qualname__} not implemented yet: {exc}')
            return default

    def _publish_status(self, detail=''):
        msg = String()
        msg.data = json.dumps({'state': self.state, 'detail': detail})
        self.pub_status.publish(msg)

    def _set_state(self, new_state, detail=''):
        """상태 전이는 전부 이 함수 하나를 거친다 - 상태 변경과 /task/status 퍼블리시가 항상 같이 일어나게 하기 위함."""
        self.state = new_state
        self._publish_status(detail)

    def _on_fault(self, msg):
        """어느 상태에 있든 /robot/fault 수신 시 즉시 FAULT로 전이 (3절 표의 공통 규칙)."""
        if self.state == State.FAULT:
            return
        self._set_state(State.FAULT, detail=msg.data)

    # ---- IDLE -> PARSING -> MOVE_TO_WATCH ----

    def _call_llm(self, text: str) -> dict:
        """LLM API를 호출해 {"tool": ..., "action": ...}를 반환한다. 스키마 검증·재시도 포함."""
        raise NotImplementedError('_call_llm 구현 필요')

    def _on_user_command(self, msg):
        """IDLE 상태에서만 명령을 받는다 - 처리 중에 새 명령이 들어와도 무시."""
        if self.state != State.IDLE:
            return
        self._set_state(State.PARSING, detail=msg.data)
        self._handle_parsing(msg.data)

    def _handle_parsing(self, text):
        parsed = self._safe_call(self._call_llm, text, default=None)
        if not parsed or 'tool' not in parsed:
            self._set_state(State.IDLE, detail='명령을 이해하지 못했습니다. 다시 말씀해주세요.')
            return
        self.current_tool = parsed['tool']
        self._detect_track_cycles = 0
        self._verify_grasp_retries = 0
        self._set_state(State.MOVE_TO_WATCH)
        # 감시 자세로 이동시키는 것과 동시에 vision을 공구 추적 모드로 켠다.
        self._set_vision_mode(SetVisionMode.Request.TRACK_TOOL, self.current_tool)
        self._send_robot_goal('move_named', named_target='watch')

    def _set_vision_mode(self, mode, tool_class=''):
        """vision_node의 /vision/set_mode 서비스를 비동기 호출한다.
        응답은 _on_set_vision_mode_response에서 받아 success를 확인한다."""
        request = SetVisionMode.Request()
        request.mode = mode
        request.tool_class = tool_class
        future = self.set_mode_client.call_async(request)
        future.add_done_callback(
            lambda f: self._on_set_vision_mode_response(f, mode))

    def _on_set_vision_mode_response(self, future, requested_mode):
        response = future.result()
        if response.success:
            return
        self.get_logger().warn(f'set_vision_mode({requested_mode}) failed: {response.message}')
        # OFF 전환 실패는 이미 성공 경로(HOME 진입 등)에서 발생하는 정리성 호출이라
        # 상태를 FAULT로 덮어쓰지 않고 경고만 남긴다. TRACK_TOOL/TRACK_HAND 전환 실패는
        # 이후 단계가 정상 동작할 수 없으므로 FAULT로 보낸다.
        if requested_mode != SetVisionMode.Request.OFF:
            self._set_state(State.FAULT, detail=f'vision 모드 전환 실패: {response.message}')

    def _send_robot_goal(self, task_type, named_target='', target_pose=None,
                          tool_class='', grasp_width_mm=0.0, grasp_force_n=0.0):
        """robot_control에 보내는 모든 RobotTask goal은 이 함수 하나를 거친다.
        task_type에 따라 의미 있는 필드만 채우고 나머지는 기본값으로 둔다(4.5절 주석 참고)."""
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
        future.add_done_callback(self._on_goal_response)

    def _on_robot_feedback(self, feedback_msg):
        """servo_pick 진행 중 robot_control이 보내는 state(tracking/descending/...)를
        그대로 /task/status에 실어 UI에 전달한다."""
        self._publish_status(detail=f'servo:{feedback_msg.feedback.state}')

    def _on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._set_state(State.FAULT, detail='goal rejected')
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_robot_result)

    def _on_robot_result(self, future):
        """모든 RobotTask result가 여기로 들어온다. goal 안에 어떤 task_type을 보냈는지는
        기억 안 하고, "지금 내가 어느 상태인가"만 보고 결과를 해석한다 - task_manager가
        한 번에 goal 하나만 진행 중이라는 전제가 이 분기의 전제조건이다."""
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
        elif self.state == State.TRACK_HAND:
            self._handle_track_hand_result(result)
        elif self.state == State.WAIT_PULL:
            self._handle_wait_pull_result(result)
        elif self.state == State.RELEASE:
            self._handle_release_result(result)
        elif self.state == State.HOME:
            self._handle_home_result(result)

    def _handle_move_to_watch_result(self, result):
        if result.success:
            self._set_state(State.DETECT_TRACK)
        else:
            self._set_state(State.FAULT, detail=result.message)

    # ---- DETECT_TRACK -> SERVO_PICK ----

    def _check_trigger(self, tool_track_msg) -> bool:
        """시야 내 + approaching이면 True (완화된 트리거 판정, 데모.md 1.3절)."""
        raise NotImplementedError('_check_trigger 구현 필요')

    def _get_grasp_spec(self, tool_class: str):
        """(grasp_width_mm, grasp_force_n) 등록된 공구 스펙을 반환한다."""
        raise NotImplementedError('_get_grasp_spec 구현 필요')

    def _on_tool_track(self, msg):
        """DETECT_TRACK 상태에서만 의미 있다 - 그 외 상태에서 온 ToolTrack은 그냥 버린다
        (SERVO_PICK 중에는 robot_control이 이 토픽을 직접 구독하지, task_manager를 거치지 않는다)."""
        if self.state != State.DETECT_TRACK:
            return
        triggered = self._safe_call(self._check_trigger, msg, default=False)
        if not triggered:
            self._detect_track_cycles += 1
            max_cycles = self.get_parameter('detect_track_max_cycles').value
            if self._detect_track_cycles >= max_cycles:
                self._set_state(State.IDLE, detail='벨트에 없음')
            return
        spec = self._safe_call(self._get_grasp_spec, self.current_tool, default=None)
        width_mm, force_n = spec if spec else (0.0, 0.0)
        self._set_state(State.SERVO_PICK)
        # 1회만 전송하고 이후는 감독만 - 좌표·제어는 robot_control의 calculate 모듈이 담당(2절).
        self._send_robot_goal(
            'servo_pick', tool_class=self.current_tool,
            grasp_width_mm=width_mm, grasp_force_n=force_n)

    # ---- SERVO_PICK -> VERIFY_GRASP -> MOVE_SAFE ----

    def _verify_grasp(self, result) -> bool:
        """무게·폭·grip_detected 삼중 확인 (데모.md 2.6/VERIFY_GRASP)."""
        raise NotImplementedError('_verify_grasp 구현 필요')

    def _handle_servo_pick_result(self, result):
        if not result.success:
            # abort 사유 문자열에 'torque'가 포함되면 충돌 등 안전 문제로 보고 즉시 FAULT,
            # 그 외(발산/추적유실/타임아웃 등)는 DETECT_TRACK으로 되돌아가 재시도.
            if 'torque' in result.message:
                self._set_state(State.FAULT, detail=result.message)
            else:
                self._detect_track_cycles = 0
                self._set_state(State.DETECT_TRACK, detail=result.message)
            return
        self._set_state(State.VERIFY_GRASP)
        verified = self._safe_call(self._verify_grasp, result, default=False)
        if verified:
            self._set_state(State.MOVE_SAFE)
            self._send_robot_goal('move_named', named_target='safe')
            return
        self._verify_grasp_retries += 1
        max_retries = self.get_parameter('verify_grasp_max_retries').value
        if self._verify_grasp_retries > max_retries:
            self._set_state(State.IDLE, detail='파지 검증 실패 - 보고')
            return
        self._send_robot_goal('release_and_retry')

    def _handle_release_and_retry_result(self, result):
        """release_and_retry의 result는 VERIFY_GRASP 상태에서 받는다(재시도 goal이므로)."""
        if result.success:
            self._detect_track_cycles = 0
            self._set_state(State.DETECT_TRACK)
        else:
            self._set_state(State.FAULT, detail=result.message)

    # ---- MOVE_SAFE -> TRACK_HAND -> WAIT_PULL ----

    def _handle_move_safe_result(self, result):
        if result.success:
            self._set_state(State.TRACK_HAND)
            self._set_vision_mode(SetVisionMode.Request.TRACK_HAND)
            timeout_s = self.get_parameter('hand_detect_timeout_s').value
            self._hand_timeout_timer = self.create_timer(timeout_s, self._on_hand_timeout)
        else:
            self._set_state(State.FAULT, detail=result.message)

    def _on_hand_timeout(self):
        """손을 못 찾고 시간 초과 -> 기본 전달 자세로 폴백."""
        self._hand_timeout_timer.cancel()
        self._hand_timeout_timer = None
        if self.state != State.TRACK_HAND:
            return
        self._send_robot_goal('move_named', named_target='handover_default')

    def _on_hand_pose(self, msg):
        """손 위치를 받으면 손 위 8cm 지점으로 이동 goal을 보낸다 - 좌표 계산(오프셋)은
        여기 task_manager 몫, robot_control은 받은 좌표로 이동만 한다(원칙대로)."""
        if self.state != State.TRACK_HAND:
            return
        if self._hand_timeout_timer is not None:
            self._hand_timeout_timer.cancel()
            self._hand_timeout_timer = None
        offset_pose = PoseStamped()
        offset_pose.header = msg.header
        offset_pose.pose = msg.pose
        offset_pose.pose.position.z += 0.08
        self._send_robot_goal('move_pose', target_pose=offset_pose)

    def _handle_track_hand_result(self, result):
        if result.success:
            self._set_state(State.WAIT_PULL)
            timeout_s = self.get_parameter('wait_pull_timeout_s').value
            self._wait_pull_timeout_timer = self.create_timer(timeout_s, self._on_wait_pull_timeout)
            self._send_robot_goal('handover_hold')
        else:
            self._set_state(State.FAULT, detail=result.message)

    # ---- WAIT_PULL -> RELEASE -> HOME ----

    def _on_wait_pull_timeout(self):
        """당김이 안 감지된 채 타임아웃 -> 내려놓고 홈으로."""
        self._wait_pull_timeout_timer.cancel()
        self._wait_pull_timeout_timer = None
        if self.state != State.WAIT_PULL:
            return
        self._set_state(State.RELEASE, detail='wait_pull timeout')
        self._send_robot_goal('place_down', named_target='place_down')

    def _handle_wait_pull_result(self, result):
        """handover_hold의 result. 성공 = 당김이 감지되어 robot_control이 이미 그리퍼를 열었다는 뜻."""
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
            self._set_state(State.FAULT, detail=result.message)

    def _handle_release_result(self, result):
        # WAIT_PULL 타임아웃 후 보낸 place_down goal의 결과 처리
        if result.success:
            self._set_state(State.HOME)
            self._set_vision_mode(SetVisionMode.Request.OFF)
            self._send_robot_goal('move_named', named_target='home')
        else:
            self._set_state(State.FAULT, detail=result.message)

    def _handle_home_result(self, result):
        if result.success:
            self._set_state(State.IDLE, detail=f'DONE tool={self.current_tool}')
        else:
            self._set_state(State.FAULT, detail=result.message)


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
