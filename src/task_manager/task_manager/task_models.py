from collections import namedtuple

from task_manager.command_parser import SUPPORTED_TOOL_CLASSES


class State:
    IDLE = 'IDLE'
    MOVE_TO_WATCH = 'MOVE_TO_WATCH'
    DETECT_TRACK = 'DETECT_TRACK'
    SERVO_PICK = 'SERVO_PICK'
    VERIFY_GRASP = 'VERIFY_GRASP'
    MOVE_SAFE = 'MOVE_SAFE'
    APPROACH_HAND = 'APPROACH_HAND'
    WAIT_PULL = 'WAIT_PULL'
    HOME = 'HOME'
    MANUAL_MOVE = 'MANUAL_MOVE'
    CANCELLING = 'CANCELLING'


class Safety:
    NORMAL = 'NORMAL'
    PROTECTIVE_STOP = 'PROTECTIVE_STOP'
    EMERGENCY_STOP = 'EMERGENCY_STOP'
    FAULT = 'FAULT'
    RECOVERY_REQUIRED = 'RECOVERY_REQUIRED'


SAFETY_PRIORITY = {
    Safety.NORMAL: 0,
    Safety.RECOVERY_REQUIRED: 1,
    Safety.PROTECTIVE_STOP: 2,
    Safety.FAULT: 3,
    Safety.EMERGENCY_STOP: 4,
}


WAIT_PULL_REMINDER_MESSAGE = '도구를 가져가세요.'

GraspSpec = namedtuple(
    'GraspSpec',
    ['width_mm', 'force_n', 'verify_min_width_mm', 'verify_max_width_mm'],
)
