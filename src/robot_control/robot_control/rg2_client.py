"""하위 호환용 재노출 모듈.

RG2Client의 실제 구현은 robot_control_node.py로 이동했다 (요청: "핵심 로직을
robot_control_node.py/servo_loop.py 두 파일 중심으로 정리"). 기존
``from robot_control.rg2_client import RG2Client`` 형태의 import가 계속
동작하도록 이 모듈에서 재노출만 한다.
"""

from robot_control.robot_control_node import RG2Client

__all__ = ['RG2Client']
