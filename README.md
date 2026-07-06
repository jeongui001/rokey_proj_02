# rokey-proj-02
[두산로보틱스 부트캠프 로키] robotics limbs

## 설치

```bash
# ROS2 패키지 의존성 (rosbridge_server, realsense2_camera 등)
rosdep install --from-paths src --ignore-src -y

# operator_gui(PyQt) pip 의존성
pip install -r src/operator_gui/requirements.txt
pip install -r src/operator_gui/test-requirements.txt  # 테스트 실행 시에만 필요

colcon build --symlink-install
```
