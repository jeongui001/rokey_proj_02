import os
import sys

from PyQt5 import QtWidgets

from operator_gui.main_window import MainWindow
from operator_gui.ros_client import RosClient


def main():
    app = QtWidgets.QApplication(sys.argv)
    host = os.environ.get('OPERATOR_GUI_ROSBRIDGE_HOST', 'localhost')
    port = int(os.environ.get('OPERATOR_GUI_ROSBRIDGE_PORT', '9090'))
    # 파싱/검증(finite 양수 확인)은 MainWindow가 담당한다 - 여기서 float()로 미리
    # 변환하면 잘못된 환경변수 하나 때문에 GUI가 시작 중 죽을 수 있다.
    camera_stale_timeout_s = os.environ.get('OPERATOR_GUI_CAMERA_STALE_TIMEOUT_S')
    ros_client = RosClient(host=host, port=port)
    window = MainWindow(ros_client, camera_stale_timeout_s=camera_stale_timeout_s)
    window.show()
    # 연결은 백그라운드 스레드에서 비동기로 시도된다 (rosbridge가 없어도 UI는 정상 기동).
    ros_client.connect()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
