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


# new_state -> (phase, checkpoint_id). 없는 전이는 파이프라인 점검.md 체크리스트 항목이 아니다.
_STATE_CHECKPOINTS = {
    State.MOVE_TO_WATCH: ('B', 'parse_no_intermediate_state'),
    State.SERVO_PICK: ('D', 'servo_pick_state_entered'),
    State.MOVE_SAFE: ('E', 'move_safe_entered'),
    State.APPROACH_HAND: ('G', 'approach_hand_entered'),
    State.WAIT_PULL: ('I', 'wait_pull_entered'),
    State.IDLE: ('K', 'idle_entered'),
}


class TaskManagerNode(Node, ActionCoordinator, SafetyRecovery, TaskFlow):
    def __init__(self):
        super().__init__('task_manager')

        self.declare_parameter('detect_track_timeout_s', 5.0)
        self.declare_parameter('verify_grasp_max_retries', 2)
        self.declare_parameter('wait_pull_timeout_s', 60.0)
        self.declare_parameter('wait_pull_reminder_interval_s', 10.0)
        self.declare_parameter('cancel_timeout_s', 5.0)
        # 응답이 없으면 RECOVERY_REQUIRED를 유지한 채 재시도를 허용한다(안전 동작은 바꾸지 않음).
        self.declare_parameter('recovery_timeout_s', 5.0)
        # 응답 없이는 다음 goal이 나가지 않으므로, 타임아웃이 없으면 vision_node 무응답 시 영구히 멈춘다.
        self.declare_parameter('vision_mode_timeout_s', 5.0)
        self.declare_parameter('status_publish_period_s', 1.0)
        # DEBUG_LOG: 실기 디버깅용 구조화 이벤트. 안정화 후 GUI/로그 정책 확정 시 제거 가능.
        self.declare_parameter('debug.publish_events', True)
        self.declare_parameter('debug.log_task_decisions', False)

        # false(기본값)면 AUTO/MANUAL 어느 쪽에서도 실제 goal을 보내지 않는다(_handle_fetch_tool).
        self.declare_parameter('auto.config_ready', False)

        # 미설정(-1/빈 문자열)이면 _check_trigger가 항상 False를 반환한다(fail-closed).
        self.declare_parameter('trigger.min_confidence', -1.0)
        self.declare_parameter('trigger.require_depth_valid', True)
        self.declare_parameter('trigger.require_approaching', True)
        self.declare_parameter('trigger.required_frame_id', '')
        self.declare_parameter('trigger.max_track_age_s', -1.0)

        # 공구별 grasp spec. 미설정(-1)이면 _get_grasp_spec이 None을 반환한다.
        for _tool in SUPPORTED_TOOL_CLASSES:
            self.declare_parameter(f'tools.{_tool}.width_mm', -1.0)
            self.declare_parameter(f'tools.{_tool}.force_n', -1.0)
            self.declare_parameter(f'tools.{_tool}.verify_min_width_mm', -1.0)
            self.declare_parameter(f'tools.{_tool}.verify_max_width_mm', -1.0)

        self.state = State.IDLE
        self.operation_mode = Mode.MANUAL
        self.safety_state = Safety.NORMAL
        self.current_tool = None
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
        self._vision_timeout_timer = None
        self._vision_timeout_owner_generation = None
        # generation 증가로 이전 Fault의 지연 복구 응답을 무시한다(_on_recover_response 참고).
        self._recovery_generation = 0
        self._recovery_in_progress = False
        self._recovery_timeout_timer = None
        self._recovery_timeout_owner_generation = None
        self._last_fault_detail = None
        # FAULT 진입 시점 상태 스냅샷 - 'continue'(파지 검증됨, 그대로 이어감) 또는
        # 'retry_pick'(그리퍼 상태 불확실, release_and_retry 후 재시작), 없으면 None.
        self._resume_kind = None
        self._resume_state = None
        self._resume_tool = None
        self._resume_grasp_spec = None
        self._last_status_detail = ''

        status_qos = QoSProfile(
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

        # _cancel_all_timers()는 이 타이머를 건드리지 않는다.
        self._status_publish_timer = self.create_timer(
            self.get_parameter('status_publish_period_s').value, self._on_status_publish_timer)

        self._publish_status(detail='')

    def _checkpoint_event(
            self, phase, checkpoint_id, status, message, data=None,
            *, throttle_s=None, log=False):
        """파이프라인 점검.md의 Phase 체크리스트에 대응하는 이벤트를 발행한다."""
        now = time.monotonic()
        key = (checkpoint_id, status)
        if throttle_s is not None:
            last = getattr(self, '_checkpoint_event_last', {}).get(key, 0.0)
            if now - last < throttle_s:
                return
            if not hasattr(self, '_checkpoint_event_last'):
                self._checkpoint_event_last = {}
            self._checkpoint_event_last[key] = now
        payload = {
            'phase': phase,
            'checkpoint_id': checkpoint_id,
            'status': status,
            'message': message,
            'data': data or {},
            'node': self.get_name(),
            'stamp_monotonic': now,
        }
        if bool(self.get_parameter('debug.publish_events').value):
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.pub_debug_events.publish(msg)
        if log or bool(self.get_parameter('debug.log_task_decisions').value):
            text = f'[CHECKPOINT][{phase}/{checkpoint_id}] status={status} message={message}'
            if status == 'FAIL':
                self.get_logger().error(text)
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
            # _handle_resume의 가드 조건과 동일 - GUI가 재개 버튼 활성화 여부에 사용.
            'resumable': (
                self._resume_kind is not None
                and self.safety_state == Safety.NORMAL
                and self.state == State.IDLE),
        }, ensure_ascii=False)
        self.pub_status.publish(msg)

    def _on_status_publish_timer(self):
        self._publish_status(self._last_status_detail)

    def _set_state(self, new_state, detail=''):
        old_state = self.state
        self.state = new_state
        self._publish_status(detail)
        if old_state is not None and old_state != new_state:
            checkpoint = _STATE_CHECKPOINTS.get(new_state)
            if checkpoint is not None:
                phase, checkpoint_id = checkpoint
                self._checkpoint_event(
                    phase, checkpoint_id, 'PASS', f'{old_state} -> {new_state}',
                    {'from': old_state, 'to': new_state, 'detail': detail})

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
