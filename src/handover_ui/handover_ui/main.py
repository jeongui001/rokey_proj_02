import sys

from PyQt5 import QtWidgets

from handover_ui.main_window import MainWindow
from handover_ui.ros_client import RosClient


def main():
    app = QtWidgets.QApplication(sys.argv)
    ros_client = RosClient(host='localhost', port=9090)
    window = MainWindow(ros_client)
    ros_client.connect()
    ros_client.subscribe_all()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
