import time

import rclpy
from std_msgs.msg import String


class SafetyState:
    NORMAL = 'NORMAL'
    PROTECTIVE_STOP = 'PROTECTIVE_STOP'
    EMERGENCY_STOP = 'EMERGENCY_STOP'
    FAULT = 'FAULT'


SAFETY_STATE_PRIORITY = {
    SafetyState.NORMAL: 0,
    SafetyState.PROTECTIVE_STOP: 1,
    SafetyState.FAULT: 2,
    SafetyState.EMERGENCY_STOP: 3,
}


class FaultPrefix:
    PROTECTIVE_STOP = 'PROTECTIVE_STOP: '
    EMERGENCY_STOP = 'EMERGENCY_STOP: '
    FAULT = 'FAULT: '


class DoosanRobotState:
    INITIALIZING = 0
    STANDBY = 1
    MOVING = 2
    SAFE_OFF = 3
    TEACHING = 4
    SAFE_STOP = 5
    EMERGENCY_STOP = 6
    HOMMING = 7
    RECOVERY = 8
    SAFE_STOP2 = 9
    SAFE_OFF2 = 10
    NOT_READY = 15


class DoosanRobotControl:
    CONTROL_INIT_CONFIG = 0
    CONTROL_ENABLE_OPERATION = 1
    CONTROL_RESET_SAFET_STOP = 2
    CONTROL_RESET_SAFET_OFF = 3
    CONTROL_RECOVERY_SAFE_STOP = 4
    CONTROL_RECOVERY_SAFE_OFF = 5
    CONTROL_RECOVERY_BACKDRIVE = 6
    CONTROL_RESET_RECOVERY = 7


class SafetyMonitor:
    """상태 폴링, Fault 단계 상승, 복구 판단을 한곳에서 관리한다."""

    def __init__(self, node):
        self.node = node
        self.state = SafetyState.NORMAL
        self.latest_robot_state = None
        self.last_fault_reason = None
        self._sample_seq = 0

    def read_robot_state(self):
        self._sample_seq += 1
        if not self.node.hardware_enabled:
            return {
                'robot_state': DoosanRobotState.STANDBY,
                'ext_torque': [0.0] * 6,
                'tool_force': [0.0] * 6,
                'received_at': time.monotonic(),
                'sample_seq': self._sample_seq,
            }
        driver = self.node._doosan
        if driver is None:
            return None
        robot_state = driver.get_robot_state()
        if robot_state is None:
            return None
        return {
            'robot_state': robot_state,
            'ext_torque': driver.get_external_torque() or [0.0] * 6,
            'tool_force': driver.get_tool_force(
                ref=self.node.get_parameter('handover_hold.ref').value) or [0.0] * 6,
            'received_at': time.monotonic(),
            'sample_seq': self._sample_seq,
        }

    def check_fault(self, sample):
        """robot_state(두산 자체 안전 상태)만 확인한다. 외력 감지는 이제 이 폴링
        경로가 아니라 DrflForceMonitor(독립 쓰레드, ROS 서비스를 거치지 않는 직접
        연결)가 전담한다 - MOVING 중에도 동작해야 해서 여기서는 뺐다(2026-07-06,
        옛 delta/baseline 방식 제거)."""
        if not isinstance(sample, dict):
            return None
        code = sample.get('robot_state')
        if code == DoosanRobotState.EMERGENCY_STOP:
            return (
                f'{FaultPrefix.EMERGENCY_STOP}물리 비상정지(E-Stop)가 감지되었습니다 '
                f'(robot_state={code}).')
        if code in (
                DoosanRobotState.SAFE_STOP,
                DoosanRobotState.SAFE_STOP2,
                DoosanRobotState.SAFE_OFF,
                DoosanRobotState.SAFE_OFF2,
        ):
            return (
                f'{FaultPrefix.PROTECTIVE_STOP}보호정지 상태가 감지되었습니다 '
                f'(robot_state={code}).')
        return None

    @staticmethod
    def classify_fault_level(reason):
        if reason.startswith(FaultPrefix.EMERGENCY_STOP):
            return SafetyState.EMERGENCY_STOP
        if reason.startswith(FaultPrefix.PROTECTIVE_STOP):
            return SafetyState.PROTECTIVE_STOP
        return SafetyState.FAULT

    def declare_fault(self, reason):
        new_state = self.classify_fault_level(reason)
        if SAFETY_STATE_PRIORITY[new_state] < SAFETY_STATE_PRIORITY[self.state]:
            return
        self.state = new_state
        self.last_fault_reason = reason
        self.node._cleanup_stop_motion()
        message = String()
        message.data = reason
        self.node.pub_fault.publish(message)
        self.node.get_logger().error(
            f'안전 fault가 선언되었습니다: state={new_state}, reason={reason}')

    def wait_for_pull(self, goal_handle, is_pull_detected, is_fresh):
        """서로 다른 최신 힘 샘플에서 연속 당김이 확인될 때까지 기다린다."""
        poll = self.node.get_parameter('handover_hold.poll_interval_s').value
        max_age = self.node.get_parameter(
            'handover_hold.force_sample_max_age_s').value
        needed = max(
            1, int(self.node.get_parameter(
                'handover_hold.pull_confirm_samples').value))
        started = time.monotonic()
        count = 0
        last_sequence = None
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                return 'CANCELED'
            if self.state != SafetyState.NORMAL:
                return 'FAULT'
            sample = self.latest_robot_state
            sequence = sample.get('sample_seq') if isinstance(sample, dict) else None
            if not is_fresh(sample, started, max_age):
                count, last_sequence = 0, None
            elif sequence is not None and sequence != last_sequence:
                last_sequence = sequence
                count = count + 1 if is_pull_detected(sample) else 0
            if count >= needed:
                return 'PULLED'
            time.sleep(poll)
        return 'SHUTDOWN'

    def recover(self, _request, response):
        if self.state == SafetyState.NORMAL:
            response.success, response.message = True, '이미 정상 상태입니다.'
            return response
        if not self.node.hardware_enabled:
            if self.state == SafetyState.EMERGENCY_STOP:
                response.success = False
                response.message = '[dry_run] 물리 E-Stop은 소프트웨어로 복구할 수 없습니다.'
            else:
                self.state = SafetyState.NORMAL
                response.success, response.message = True, '[dry_run] 복구되었습니다.'
            return response

        driver = self.node._doosan
        if driver is None:
            response.success, response.message = False, 'DoosanDriver가 없습니다.'
            return response
        robot_state = driver.get_robot_state()
        if robot_state == DoosanRobotState.EMERGENCY_STOP:
            response.success = False
            response.message = '물리 E-Stop이 눌려 있어 소프트웨어 복구를 거절합니다.'
            return response
        if robot_state == DoosanRobotState.STANDBY:
            torque = driver.get_external_torque()
            if torque is None or len(torque) != 6:
                response.success, response.message = False, '외력 측정 실패 - 복구를 보류합니다.'
                return response
            # DrflForceMonitor와 같은 관절별 절대 임계값(direct_threshold_nm)을
            # 재사용한다 - baseline/delta 방식은 제거됨(2026-07-06). 지금 이 순간
            # 외력이 그 기준을 넘지 않아야만 정상 복구로 인정한다.
            thresholds = self.node.get_parameter(
                'safety.external_torque.direct_threshold_nm').value
            safe = all(abs(value) <= threshold for value, threshold in zip(torque, thresholds))
            if safe:
                self.state = SafetyState.NORMAL
            response.success = safe
            response.message = (
                '복구 완료' if safe else '복구 조건 미충족 - 외력이 여전히 높습니다.')
            return response

        control = {
            DoosanRobotState.SAFE_STOP:
                DoosanRobotControl.CONTROL_RESET_SAFET_STOP,
            DoosanRobotState.SAFE_OFF:
                DoosanRobotControl.CONTROL_RESET_SAFET_OFF,
            DoosanRobotState.SAFE_STOP2:
                DoosanRobotControl.CONTROL_RECOVERY_SAFE_STOP,
            DoosanRobotState.SAFE_OFF2:
                DoosanRobotControl.CONTROL_RECOVERY_SAFE_OFF,
        }.get(robot_state)
        if control is None:
            response.success = False
            response.message = f'복구할 수 없는 상태입니다 (robot_state={robot_state}).'
            return response

        driver.set_robot_control(control)
        new_state = driver.get_robot_state()
        if new_state == DoosanRobotState.STANDBY:
            self.state = SafetyState.NORMAL
            response.success = True
            response.message = f'복구 완료 ({robot_state} -> {new_state})'
        elif new_state == DoosanRobotState.RECOVERY:
            response.success = False
            response.message = 'RECOVERY 상태입니다. CONTROL_RESET_RECOVERY 확인이 필요합니다.'
        else:
            response.success = False
            response.message = f'복구 실패 (robot_state={new_state}).'
        return response
