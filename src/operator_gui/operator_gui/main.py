import os
import sys

from PyQt5 import QtWidgets

from operator_gui.main_window import MainWindow
from operator_gui.ros_client import RosClient


def main():
    app = QtWidgets.QApplication(sys.argv)
    host = os.environ.get('OPERATOR_GUI_ROSBRIDGE_HOST', 'localhost')
    port = int(os.environ.get('OPERATOR_GUI_ROSBRIDGE_PORT', '9090'))
    ros_client = RosClient(host=host, port=port)
    window = MainWindow(ros_client)
    window.show()
    # 연결은 백그라운드 스레드에서 비동기로 시도된다 (rosbridge가 없어도 UI는 정상 기동).
    ros_client.connect()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
