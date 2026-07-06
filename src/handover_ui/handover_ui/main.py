import sys

import rclpy
from PyQt5 import QtWidgets

from handover_ui.main_window import MainWindow
from handover_ui.ros_client import RosClient


def main():
    rclpy.init()
    app = QtWidgets.QApplication(sys.argv)
    ros_client = RosClient()
    # MainWindow 생성자가 ros_client.on_* 콜백을 먼저 꽂아두므로, 그 다음에
    # connect()/subscribe_all()을 호출해야 메시지를 놓치지 않는다.
    window = MainWindow(ros_client)
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
