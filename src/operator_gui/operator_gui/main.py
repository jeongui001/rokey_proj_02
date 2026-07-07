import os
import sys

import rclpy
from PyQt5 import QtWidgets

from operator_gui.main_window import MainWindow
from operator_gui.ros_client import RosClient


def main():
    rclpy.init()
    app = QtWidgets.QApplication(sys.argv)
    # 파싱/검증(finite 양수 확인)은 MainWindow가 담당한다 - 여기서 float()로 미리
    # 변환하면 잘못된 환경변수 하나 때문에 GUI가 시작 중 죽을 수 있다.
    camera_stale_timeout_s = os.environ.get('OPERATOR_GUI_CAMERA_STALE_TIMEOUT_S')
    ros_client = RosClient()
    # MainWindow 생성자가 ros_client.on_* 콜백을 먼저 꽂아두므로, 그 다음에
    # connect()/subscribe_all()을 호출해야 메시지를 놓치지 않는다.
    window = MainWindow(ros_client, camera_stale_timeout_s=camera_stale_timeout_s)
    ros_client.connect()
    ros_client.subscribe_all()
    window.show()

    def _shutdown():
        ros_client.close()
        rclpy.shutdown()

    app.aboutToQuit.connect(_shutdown)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
