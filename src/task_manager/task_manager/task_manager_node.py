import json
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from handover_interfaces.action import RobotTask
from handover_interfaces.msg import ToolTrack
from handover_interfaces.srv import SetVisionMode

from task_manager.action_coordinator import ActionCoordinator
from task_manager.command_parser import Mode
from task_manager.safety_recovery import SafetyRecovery
from task_manager.task_flow import TaskFlow
from task_manager.task_models import (
    GraspSpec,
    Safety,
    State,
    SUPPORTED_TOOL_CLASSES,
    WAIT_PULL_REMINDER_MESSAGE,
)


class TaskManagerNode(Node, ActionCoordinator, SafetyRecovery, TaskFlow):
    def __init__(self):
        super().__init__('task_manager')

        self.declare_parameter('detect_track_timeout_s', 5.0)
        self.declare_parameter('verify_grasp_max_retries', 2)
        self.declare_parameter('wait_pull_timeout_s', 60.0)
        # wait_pull_timeout_s가 지난 뒤에도 handover_hold를 취소하지 않고 계속 들고
        # 대기하면서, 이 간격으로 GUI 안내(WAIT_PULL_REMINDER_MESSAGE)를 반복
        # 재발행한다 (_on_wait_pull_timeout/_on_wait_pull_reminder 참고).
        self.declare_parameter('wait_pull_reminder_interval_s', 10.0)
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
        # DEBUG_LOG: 실기 디버깅용 구조화 이벤트. 안정화 후 GUI/로그 정책 확정 시 제거 가능.
        self.declare_parameter('debug.publish_events', True)
        self.declare_parameter('debug.log_task_decisions', False)

        # AUTO 설정 준비 게이트 - false(기본값)면 AUTO/MANUAL 모드 어느 쪽에서도
        # 실제 물체 가져오기 goal은 보내지 않는다 (_handle_fetch_tool 참고).
        # fetch_tool 자체는 AUTO/MANUAL 모드 구분 없이 동작한다 - MANUAL은 여기에
        # 개별 이동 명령(_handle_manual_move)을 추가로 더 허용할 뿐이다.
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
        self._goal_result_state = None
        self._cancel_pending_callback = None
        self._vision_generation = 0
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
        # "재개" 스냅샷 - FAULT 진입 시점의 상태를 저장해 두었다가, 복구(NORMAL) 후
        # 명시적 "재개" 명령이 오면 이어서 진행할지 판단한다 (_capture_resume_snapshot/
        # _handle_resume 참고). _resume_kind는 'continue'(MOVE_SAFE/APPROACH_HAND/
        # WAIT_PULL - 파지가 이미 검증된 뒤라 그대로 이어감) 또는 'retry_pick'
        # (SERVO_PICK/VERIFY_GRASP - 그리퍼 상태가 불확실하므로 release_and_retry로
        # 안전하게 열고 DETECT_TRACK부터 다시 시작) 중 하나이거나, 재개할 게 없으면
        # None이다.
        self._resume_kind = None
        self._resume_state = None
        self._resume_tool = None
        self._resume_grasp_spec = None
        # 가장 최근에 발행한 detail - 상태 재발행 타이머가 state/detail을 임의로
        # 바꾸지 않고 마지막 값을 그대로 다시 내보내기 위해 저장해 둔다.
        self._last_status_detail = ''

        # 늦게 붙는 구독자도 마지막 상태를 즉시 받을 수 있도록 주기 재발행과
        # transient_local을 함께 설정한다.
        status_qos = QoSProfile(
            # depth=1: transient_local 구독자가 늦게 붙어도 가장 최근 상태 1개만
            # 재생하도록 한다 - 여러 개의 과거 상태가 한꺼번에 재생되지 않게 한다.
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_status = self.create_publisher(String, '/task/status', status_qos)
        self.pub_debug_events = self.create_publisher(String, '/debug/events', 10)
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

    def _debug_event(
            self, level, category, reason, message, data=None,
            *, throttle_s=None, log=False):
        """DEBUG_LOG: GUI의 '오류 확인' 패널이 모을 수 있는 최근 판단/오류 이벤트."""
        now = time.monotonic()
        key = (level, category, reason)
        if throttle_s is not None:
            last = getattr(self, '_debug_event_last', {}).get(key, 0.0)
            if now - last < throttle_s:
                return
            if not hasattr(self, '_debug_event_last'):
                self._debug_event_last = {}
            self._debug_event_last[key] = now
        payload = {
            'node': self.get_name(),
            'level': level,
            'category': category,
            'reason': reason,
            'message': message,
            'data': data or {},
            'stamp_monotonic': now,
        }
        if bool(self.get_parameter('debug.publish_events').value):
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.pub_debug_events.publish(msg)
        if log or bool(self.get_parameter('debug.log_task_decisions').value):
            text = (
                f'[TASK][{category}] level={level} reason={reason} '
                f'message={message} data={payload["data"]}')
            if level in ('ERROR', 'FAULT'):
                self.get_logger().error(text)
            elif level == 'WARN':
                self.get_logger().warn(text)
            else:
                self.get_logger().info(text)

    def _publish_status(self, detail=''):
        self._last_status_detail = detail
        msg = String()
        msg.data = json.dumps({
            'state': self.state,
            'detail': detail,
            'operation_mode': self.operation_mode,
            'safety_state': self.safety_state,
            # GUI가 "재개" 버튼을 정확히 언제 활성화할지 판단하는 데 쓴다 - 재개할
            # 스냅샷이 있고, 지금 당장 안전하게 재개를 시작할 수 있는 상태일 때만
            # true다 (_handle_resume의 실제 가드와 동일한 조건).
            'resumable': (
                self._resume_kind is not None
                and self.safety_state == Safety.NORMAL
                and self.state == State.IDLE),
        }, ensure_ascii=False)
        self.pub_status.publish(msg)

    def _on_status_publish_timer(self):
        """주기적으로 최신 상태를 재발행한다 - 늦게 연결된 GUI도 다음 주기 안에
        현재 상태를 받을 수 있게 한다. state/detail을 새로 바꾸지 않고 마지막
        detail을 그대로 다시 내보낸다."""
        self._publish_status(self._last_status_detail)

    def _set_state(self, new_state, detail=''):
        old_state = self.state
        self.state = new_state
        self._publish_status(detail)
        if old_state != new_state:
            self._debug_event(
                'INFO', 'STATE', 'state_transition', 'task state changed',
                {'from': old_state, 'to': new_state, 'detail': detail},
                log=False)

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

def main(args=None):
    rclpy.init(args=args)
    node = TaskManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
