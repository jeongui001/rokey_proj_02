import json
import math
from collections import namedtuple

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

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
    EMERGENCY_STOP = 'EMERGENCY_STOP'
    FAULT = 'FAULT'
    # "리셋" 명령을 받았다는 사실만으로 안전상태를 NORMAL로 되돌리지 않는다.
    # robot_control의 /robot/recover(std_srvs/Trigger)가 success=true를 반환해야만
    # NORMAL로 전환한다 (TaskManagerNode._on_recover_response 참고).
    RECOVERY_REQUIRED = 'RECOVERY_REQUIRED'


# 안전상태 우선순위: 숫자가 클수록 더 심각하다. robot_control.robot_control_node의
# SAFETY_STATE_PRIORITY와 합의된 순서(task_manager에만 있는 RECOVERY_REQUIRED는
# NORMAL 바로 위, PROTECTIVE_STOP 바로 아래에 위치)와 일치시킨다.
SAFETY_PRIORITY = {
    Safety.NORMAL: 0,
    Safety.RECOVERY_REQUIRED: 1,
    Safety.PROTECTIVE_STOP: 2,
    Safety.FAULT: 3,
    Safety.EMERGENCY_STOP: 4,
}

# command_parser._TOOL_KEYWORDS의 값과 반드시 일치해야 하는 tool_class 목록.
SUPPORTED_TOOL_CLASSES = ('water_bottle', 'spanner', 'driver', 'wrench', 'pliers', 'hammer')

# _get_grasp_spec()이 반환하는 공구별 grasp 스펙. servo_pick에 보낼 값(width_mm,
# force_n)과 VERIFY_GRASP에서 결과를 검증할 때 쓰는 값(verify_min/max_width_mm,
# payload_min/max_kg)을 함께 묶어, servo_pick을 보낼 때 사용한 스펙을 그대로
# 검증에도 재사용할 수 있게 한다 (_on_tool_track/_verify_grasp 참고).
GraspSpec = namedtuple('GraspSpec', [
    'width_mm', 'force_n', 'verify_min_width_mm', 'verify_max_width_mm',
    'payload_min_kg', 'payload_max_kg',
])


class TaskManagerNode(Node):
    def __init__(self):
        super().__init__('task_manager')

        self.declare_parameter('detect_track_timeout_s', 5.0)
        self.declare_parameter('verify_grasp_max_retries', 2)
        self.declare_parameter('wait_pull_timeout_s', 60.0)
        self.declare_parameter('cancel_timeout_s', 5.0)
        # /robot/recover 응답을 무한정 기다리지 않기 위한 순수 통신 타임아웃이다.
        # 이 기본값은 안전 동작(언제 NORMAL로 전환하는지 등)을 바꾸지 않는다 - 응답이
        # 오지 않을 때 RECOVERY_REQUIRED를 유지한 채 재시도를 허용할 뿐이다. 실제
        # 환경(네트워크/서비스 응답 지연)에 맞게 조정 가능하다.
        self.declare_parameter('recovery_timeout_s', 5.0)
        # 초기 연결 또는 재연결(늦게 연결된 GUI)에서도 다음 주기 안에 현재 상태를
        # 받을 수 있도록 /task/status를 주기적으로 재발행한다. 순수 통신 목적의
        # 값이며, 이 자체가 state/detail을 바꾸지는 않는다 (_on_status_publish_timer).
        self.declare_parameter('status_publish_period_s', 1.0)

        # AUTO 설정 준비 게이트 - false(기본값)면 AUTO 모드 전환 자체는 되지만
        # 실제 물체 가져오기 goal은 보내지 않는다 (_handle_fetch_tool 참고).
        self.declare_parameter('auto.config_ready', False)

        # DETECT_TRACK 트리거 조건. 실제 캘리브레이션 전에는 -1/빈 문자열(미설정)로
        # 두어 _check_trigger가 항상 False를 반환하도록 한다 (fail-closed).
        self.declare_parameter('trigger.min_confidence', -1.0)
        self.declare_parameter('trigger.require_depth_valid', True)
        self.declare_parameter('trigger.require_approaching', True)
        self.declare_parameter('trigger.required_frame_id', '')
        self.declare_parameter('trigger.max_track_age_s', -1.0)

        # 공구별 grasp spec. 미설정(-1) 상태에서는 _get_grasp_spec이 None을 반환해
        # servo_pick goal을 보내지 않는다 (_on_tool_track 참고).
        for _tool in SUPPORTED_TOOL_CLASSES:
            self.declare_parameter(f'tools.{_tool}.width_mm', -1.0)
            self.declare_parameter(f'tools.{_tool}.force_n', -1.0)
            self.declare_parameter(f'tools.{_tool}.verify_min_width_mm', -1.0)
            self.declare_parameter(f'tools.{_tool}.verify_max_width_mm', -1.0)
            self.declare_parameter(f'tools.{_tool}.payload_min_kg', -1.0)
            self.declare_parameter(f'tools.{_tool}.payload_max_kg', -1.0)

        self.state = State.IDLE
        self.operation_mode = Mode.MANUAL
        self.safety_state = Safety.NORMAL
        self.current_tool = None
        # servo_pick goal 전송에 사용한 grasp spec - VERIFY_GRASP에서 동일한 spec으로
        # 결과를 검증하기 위해 저장해 둔다 (_on_tool_track에서 채워짐).
        self._active_grasp_spec = None
        self._verify_grasp_retries = 0
        self._wait_pull_timeout_timer = None
        self._detect_track_timer = None
        self._cancel_timeout_timer = None
        self._goal_in_progress = False
        self._current_goal_handle = None
        self._goal_generation = 0
        self._cancel_pending_callback = None
        # /robot/recover 복구 요청 관련 상태. generation은 새 Fault가 들어오면
        # 증가시켜, 그 이전에 보낸 복구 요청의 지연 응답이 나중에 도착해도
        # 무시하도록 한다 (_on_recover_response 참고).
        self._recovery_generation = 0
        self._recovery_in_progress = False
        # /robot/recover call_async 이후 응답 대기 타임아웃 (one-shot 성격의 재사용
        # 타이머). _on_recovery_timeout 참고. _recovery_timeout_owner_generation은
        # 현재 살아있는 타이머가 어느 generation 소유인지 기록해, 오래된 응답/타임아웃
        # 콜백이 이후 세대의 새 타이머를 실수로 취소하지 못하게 한다.
        self._recovery_timeout_timer = None
        self._recovery_timeout_owner_generation = None
        # 완전히 동일한 Fault 메시지의 반복 발행을 dedup하기 위한 마지막 detail.
        self._last_fault_detail = None
        # 가장 최근에 발행한 detail - 상태 재발행 타이머가 state/detail을 임의로
        # 바꾸지 않고 마지막 값을 그대로 다시 내보내기 위해 저장해 둔다.
        self._last_status_detail = ''

        # rosbridge(WebSocket) 경유 구독은 QoS를 신뢰할 수 없으므로 주기 재발행이
        # 필수지만, 네이티브 ROS2 구독자를 위한 방어선으로 transient_local도 함께
        # 설정한다 (늦게 붙는 구독자가 마지막 값을 즉시 받을 수 있다).
        status_qos = QoSProfile(
            # depth=1: transient_local 구독자가 늦게 붙어도 가장 최근 상태 1개만
            # 재생하도록 한다 - 여러 개의 과거 상태가 한꺼번에 재생되지 않게 한다.
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_status = self.create_publisher(String, '/task/status', status_qos)
        self.sub_command = self.create_subscription(
            String, '/user_command/text', self._on_user_command, 10)
        self.sub_tool_track = self.create_subscription(
            ToolTrack, '/vision/tool_track', self._on_tool_track, 10)
        self.sub_fault = self.create_subscription(
            String, '/robot/fault', self._on_fault, 10)

        self.set_mode_client = self.create_client(SetVisionMode, '/vision/set_mode')
        self.recover_client = self.create_client(Trigger, '/robot/recover')
        self.robot_task_client = ActionClient(self, RobotTask, 'robot_task')

        # 업무 타이머(WAIT_PULL/DETECT_TRACK)나 취소/복구 타임아웃 타이머와는 별개의
        # 카테고리다 - _cancel_all_timers()는 이 타이머를 건드리지 않는다.
        self._status_publish_timer = self.create_timer(
            self.get_parameter('status_publish_period_s').value, self._on_status_publish_timer)

        # 초기화가 끝난 뒤 초기 상태(IDLE/MANUAL/NORMAL)를 즉시 한 번 발행한다 -
        # 초기 연결된 GUI가 첫 상태를 기다리지 않게 한다.
        self._publish_status(detail='')

    def _safe_call(self, fn, *args, default=None, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            self.get_logger().warn(f'{fn.__qualname__} not implemented yet: {exc}')
            return default

    def _publish_status(self, detail=''):
        self._last_status_detail = detail
        msg = String()
        msg.data = json.dumps({
            'state': self.state,
            'detail': detail,
            'operation_mode': self.operation_mode,
            'safety_state': self.safety_state,
        }, ensure_ascii=False)
        self.pub_status.publish(msg)

    def _on_status_publish_timer(self):
        """주기적으로 최신 상태를 재발행한다 - 늦게 연결된 GUI도 다음 주기 안에
        현재 상태를 받을 수 있게 한다. state/detail을 새로 바꾸지 않고 마지막
        detail을 그대로 다시 내보낸다."""
        self._publish_status(self._last_status_detail)

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
        self._active_grasp_spec = None
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
        # 취소 타임아웃이면 복구를 시작하지 않는다 - 진행 중이던 복구 요청이 있었다면 중단한다.
        self._recovery_generation += 1
        self._recovery_in_progress = False
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
        self._goal_in_progress = False
        self._current_goal_handle = None
        self._stop_cancel_timeout_timer()
        self._cancel_pending_callback = None
        self._goal_generation += 1
        self.safety_state = Safety.FAULT
        self._set_vision_mode(SetVisionMode.Request.OFF)
        self._publish_status(
            detail=f'{context_label} 예외: {exc} - 안전을 위해 FAULT로 전환합니다.')

    # ---- 안전/고장 처리 ----

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
        # 새로운 Fault는 복구 진행보다 항상 우선한다 - 진행 중이던 복구 요청(취소
        # 대기, 서비스 응답 대기, 또는 응답 타임아웃 타이머)이 있었다면 모두
        # 무효화한다. generation을 올려두면 이미 보낸 /robot/recover 요청의 지연
        # 응답(성공이든 타임아웃이든)이 나중에 도착해도 무시된다.
        self._recovery_generation += 1
        self._recovery_in_progress = False
        self._stop_recovery_timeout_timer()
        self.safety_state = safety_state
        self._publish_status(detail=detail)
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
        future = self.recover_client.call_async(Trigger.Request())
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
        if not bool(self.get_parameter('auto.config_ready').value):
            # AUTO 모드 전환 자체는 허용하되, 실기에서 미합의 trigger/grasp spec
            # 값으로 추측 동작하지 않도록 실제 goal 송신은 막는다.
            self._publish_status(
                detail='AUTO 설정값 미확정(auto.config_ready=false) - '
                       '물체 가져오기 명령을 실행하지 않습니다.')
            return
        if self.state != State.IDLE:
            return
        self._set_state(State.PARSING, detail=f'tool={tool}')
        self.current_tool = tool
        self._active_grasp_spec = None
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
        try:
            future = self.robot_task_client.send_goal_async(
                goal, feedback_callback=self._on_robot_feedback)
        except Exception as exc:
            self._handle_action_future_exception(exc, f'{task_type} send_goal_async')
            return
        future.add_done_callback(lambda f: self._on_goal_response(f, generation))

    def _on_robot_feedback(self, feedback_msg):
        self._publish_status(detail=f'servo:{feedback_msg.feedback.state}')

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
            self._maybe_start_recovery()
            return
        try:
            response = future.result()
        except Exception as exc:
            self._handle_action_future_exception(exc, 'robot result')
            return
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
        payload_min_kg = float(self.get_parameter(f'{prefix}.payload_min_kg').value)
        payload_max_kg = float(self.get_parameter(f'{prefix}.payload_max_kg').value)

        if not all(math.isfinite(v) for v in (
                width_mm, force_n, verify_min_width_mm, verify_max_width_mm,
                payload_min_kg, payload_max_kg)):
            return None  # NaN/Inf가 하나라도 있으면 신뢰할 수 없다

        if width_mm <= 0.0 or force_n <= 0.0:
            return None  # 미설정 - 추측값으로 servo_pick을 보내지 않는다
        if verify_min_width_mm < 0.0 or verify_max_width_mm <= 0.0:
            return None
        if verify_min_width_mm > verify_max_width_mm:
            return None  # 설정 오류

        # payload_min/max_kg는 둘 다 음수(sentinel)일 때만 검증을 비활성화한다.
        # 둘 중 하나만 설정된 경우는 조용히 비활성화하지 않고 설정 오류로 취급한다.
        if payload_min_kg < 0.0 and payload_max_kg < 0.0:
            payload_min_kg = -1.0
            payload_max_kg = -1.0
        elif payload_min_kg < 0.0 or payload_max_kg < 0.0:
            return None  # 하나만 설정됨 - 설정 오류
        elif payload_min_kg > payload_max_kg:
            return None  # 설정 오류

        return GraspSpec(
            width_mm=width_mm, force_n=force_n,
            verify_min_width_mm=verify_min_width_mm, verify_max_width_mm=verify_max_width_mm,
            payload_min_kg=payload_min_kg, payload_max_kg=payload_max_kg)

    def _on_tool_track(self, msg):
        if self.state != State.DETECT_TRACK:
            return
        triggered = self._safe_call(self._check_trigger, msg, default=False)
        if not triggered:
            return
        self._cancel_detect_track_timer()
        spec = self._safe_call(self._get_grasp_spec, self.current_tool, default=None)
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
        """grip_detected/final_width_mm/(선택적)payload로 파지 성공 여부를 검증한다.

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

        payload_verification_enabled = spec.payload_min_kg >= 0.0 and spec.payload_max_kg >= 0.0
        if payload_verification_enabled:
            if not math.isfinite(result.measured_payload_kg):
                return False
            if not (spec.payload_min_kg <= result.measured_payload_kg <= spec.payload_max_kg):
                return False

        return True

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
            detail = f'DONE tool={self.current_tool}'
            self.current_tool = None
            self._active_grasp_spec = None
            self._set_state(State.IDLE, detail=detail)
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
