# rokey-proj-02
[두산로보틱스 부트캠프 로키] robotics limbs

## 사전 준비

`hardware_enabled=true`로 실제 M0609를 제어하려면(dry_run에서는 불필요) `doosan-robot2`
(dsr_msgs2 ROS2 패키지 + DRFL 라이브러리)를 별도 워크스페이스에 빌드해 source해야 한다 —
이 저장소에는 포함되어 있지 않다(`robot_control/package.xml` 참고).

- `robot_control_node`의 `safety.external_torque.drfl_lib_path` 기본값이
  `~/cobot_ws/install/dsr_hardware2/lib/libdsr_hardware2.so`를 가정한다 — 본인
  워크스페이스 이름이 `cobot_ws`가 아니면 `local_params_file`로 오버라이드해야 한다
  (`doosan_driver.controller_name`과 같은 방식, `robot_control_local_params.yaml` 참고).
- doosan-robot2 포크/버전에 따라 `dsr_controller2` 서비스/토픽 이름 앞에 붙는
  세그먼트가 다를 수 있다 — `docs/실행방식.md`의 "ros2 jazzy" 항목 참고.

## 설치

```bash
# ROS2 패키지 의존성 (realsense2_camera 등)
rosdep install --from-paths src --ignore-src -y

# 각 패키지 pip 의존성
pip install -r src/operator_gui/requirements.txt
pip install -r src/operator_gui/test-requirements.txt  # 테스트 실행 시에만 필요
pip install -r src/robot_control/requirements.txt
pip install -r src/stt_node/requirements.txt
pip install -r src/vision_node/requirements.txt

colcon build --symlink-install
```

노드별 실행 방법은 `docs/실행방식.md` 참고 (mediapipe 손 추적은 Docker 컨테이너로 별도 실행 필요).
