import base64
import json
import os
import threading

import roslibpy

DEFAULT_CAMERA_TOPIC = '/vision/debug_image/compressed'
DEFAULT_RECONNECT_INTERVAL_S = 5.0
DEFAULT_CONNECT_TIMEOUT_S = 5.0


class RosClient:
    """rosbridge(WebSocket)에 접속해 필요한 토픽을 구독/퍼블리시하는 래퍼.

    PyQt에 의존하지 않는다 - 연결/메시지 이벤트는 평범한 콜백 속성(on_*)으로만
    알리고, Qt Signal로 옮기는 것은 MainWindow의 책임이다.
    """

    def __init__(self, host='localhost', port=9090, camera_topic=None,
                 reconnect_interval_s=DEFAULT_RECONNECT_INTERVAL_S):
        self.host = host
        self.port = port
        self.camera_topic = camera_topic or os.environ.get(
            'OPERATOR_GUI_CAMERA_TOPIC', DEFAULT_CAMERA_TOPIC)
        self.reconnect_interval_s = reconnect_interval_s

        self.on_task_status = None       # (state, detail, operation_mode, safety_state)
        self.on_gripper_state = None      # (width_mm, grip_detected)
        self.on_fault = None              # (message)
        self.on_camera_image = None       # (image_bytes)
        self.on_connection_changed = None  # (is_connected: bool)

        self.ros = roslibpy.Ros(host=host, port=port)
        self._command_topic = None
        self._subscribed_topics = []  # 구독 중인 roslibpy.Topic 목록 (중복 구독 방지용)
        # ROS(WebSocket) 연결 여부와 토픽 구독 준비 여부는 서로 다른 상태다 - rosbridge
        # 연결 자체는 살아있어도(is_connected=True) error 이벤트 등으로 구독만 끊길 수
        # 있으므로 별도로 추적한다. 모든 구독 + command 토픽 생성이 성공했을 때만 True.
        self._subscriptions_ready = False
        self._connecting = False
        self._closed = False
        self._connect_lock = threading.Lock()
        self._subscribe_lock = threading.Lock()

        self.ros.on_ready(self._on_ready, run_in_thread=True)
        self.ros.on('close', self._on_close)
        self.ros.on('error', self._on_error)

    # ---- 연결 ----

    def connect(self):
        """비동기로 연결을 시도한다 (호출 스레드/UI 스레드를 막지 않는다)."""
        self._closed = False
        with self._connect_lock:
            if self._connecting:
                return
            self._connecting = True
        threading.Thread(target=self._connect_worker, daemon=True).start()

    def _connect_worker(self):
        try:
            self.ros.run(timeout=DEFAULT_CONNECT_TIMEOUT_S)
        except Exception:
            # 연결 실패 - UI는 계속 동작해야 하므로 예외를 삼키고 상태만 통지한다.
            # on_ready가 나중에라도 호출되면 그때 연결됨으로 통지된다.
            if self.on_connection_changed is not None:
                self.on_connection_changed(False)
        finally:
            self._connecting = False

    def ensure_connected(self):
        """연결과 구독 준비 상태를 확인해 필요한 조치를 취한다. UI 쪽 타이머가
        주기적으로 호출한다.

        - ROS(WebSocket) 연결 자체가 끊겨 있으면 connect()로 재연결을 시도한다.
        - 연결은 되어 있는데 구독이 준비되지 않았다면(_subscriptions_ready=False,
          예: error 이벤트로 구독만 정리된 경우) subscribe_all()을 재시도한다.
          재구독에 성공했을 때만 UI에 연결 성공을 알린다.
        - 연결과 구독이 모두 준비된 상태면 아무 것도 하지 않는다.
        """
        if self._closed:
            return
        if not self.is_connected():
            self.connect()
            return
        if self._subscriptions_ready:
            return
        if self.subscribe_all() and self.on_connection_changed is not None:
            self.on_connection_changed(True)

    def close(self):
        self._closed = True
        self._teardown_subscriptions()
        try:
            self.ros.terminate()
        except Exception:
            pass

    def is_connected(self) -> bool:
        try:
            return bool(self.ros.is_connected)
        except Exception:
            return False

    def _on_ready(self):
        success = self.subscribe_all()
        if self.on_connection_changed is not None:
            self.on_connection_changed(success)
        # 실패했다면 여기서 재시도하지 않는다 - _subscriptions_ready가 False로 남아
        # 있으므로 QTimer가 다음에 ensure_connected()를 호출할 때 다시 시도된다.

    def _on_close(self, *_args):
        # 연결이 끊겼으므로 기존 구독은 더 이상 유효하지 않다 - 정리해서, 재연결
        # 성공(on_ready) 시 subscribe_all이 토픽마다 정확히 한 번씩 다시 구독하게 한다.
        self._teardown_subscriptions()
        if self.on_connection_changed is not None:
            self.on_connection_changed(False)

    def _on_error(self, *_args):
        # 오류 이후에도 유령 구독이 남아있지 않도록 close와 동일하게 정리한다.
        # (오류가 실제 연결 종료를 동반하지 않는 경우에도, 다음 on_ready에서
        # subscribe_all이 처음부터 깨끗하게 다시 구독하도록 하는 것이 더 안전하다.)
        self._teardown_subscriptions()
        if self.on_connection_changed is not None:
            self.on_connection_changed(False)

    # ---- 구독/발행 ----

    def subscribe_all(self) -> bool:
        """필요한 토픽을 구독한다. 이미 구독이 준비되어 있으면(on_ready/
        ensure_connected가 여러 번 호출돼도) 아무 것도 하지 않고 True를 반환한다.

        구독 도중 일부 토픽에서 예외가 발생하면, 이번 시도에서 이미 만든 구독을
        정리하고 내부 상태를 빈 채로 되돌려 False를 반환한다 (예외 자체는
        호출측으로 전파하지 않는다) - 다음 시도가 처음부터 깨끗하게 재구독할 수 있다.

        반환값: 구독이 준비된 상태(이미 준비돼 있었거나 새로 전부 성공)면 True,
        일부라도 실패했으면 False.
        """
        with self._subscribe_lock:
            if self._subscriptions_ready:
                return True
            specs = (
                ('/task/status', 'std_msgs/String', self._on_task_status_raw),
                ('/gripper/state', 'handover_interfaces/GripperState', self._on_gripper_state_raw),
                ('/robot/fault', 'std_msgs/String', self._on_fault_raw),
                (self.camera_topic, 'sensor_msgs/CompressedImage', self._on_camera_image_raw),
            )
            created = []
            try:
                for name, msg_type, callback in specs:
                    topic = roslibpy.Topic(self.ros, name, msg_type)
                    topic.subscribe(callback)
                    created.append(topic)
                command_topic = roslibpy.Topic(
                    self.ros, '/user_command/text', 'std_msgs/String')
            except Exception:
                for topic in created:
                    try:
                        topic.unsubscribe()
                    except Exception:
                        pass
                self._subscribed_topics = []
                self._command_topic = None
                self._subscriptions_ready = False
                return False
            self._subscribed_topics = created
            self._command_topic = command_topic
            self._subscriptions_ready = True
            return True

    def _teardown_subscriptions(self):
        """구독 중인 토픽을 unsubscribe하고, command 토픽은 unadvertise한 뒤 내부
        목록/준비 상태를 정리한다. 개별 실패가 있어도 UI(호출측)가 죽지 않도록
        예외를 삼킨다."""
        with self._subscribe_lock:
            topics = self._subscribed_topics
            self._subscribed_topics = []
            command_topic = self._command_topic
            self._command_topic = None
            self._subscriptions_ready = False
        for topic in topics:
            try:
                topic.unsubscribe()
            except Exception:
                pass
        if command_topic is not None:
            try:
                command_topic.unadvertise()
            except Exception:
                pass

    def publish_command(self, text: str) -> bool:
        """/user_command/text로 명령을 보낸다. 연결이 없거나 빈 명령이면 보내지 않고
        False를 반환한다 (호출측이 로그에 이유를 남길 수 있도록)."""
        text = (text or '').strip()
        if not text:
            return False
        if not self.is_connected() or self._command_topic is None:
            return False
        self._command_topic.publish(roslibpy.Message({'data': text}))
        return True

    # ---- 콜백 ----

    def _on_task_status_raw(self, message):
        if self.on_task_status is None:
            return
        try:
            payload = json.loads(message.get('data', '{}'))
        except (ValueError, TypeError):
            return
        self.on_task_status(
            payload.get('state', ''),
            payload.get('detail', ''),
            payload.get('operation_mode', ''),
            payload.get('safety_state', ''))

    def _on_gripper_state_raw(self, message):
        if self.on_gripper_state is None:
            return
        self.on_gripper_state(message.get('width_mm', 0.0), message.get('grip_detected', False))

    def _on_fault_raw(self, message):
        if self.on_fault is None:
            return
        self.on_fault(message.get('data', ''))

    def _on_camera_image_raw(self, message):
        if self.on_camera_image is None:
            return
        data_b64 = message.get('data')
        if not data_b64:
            return
        try:
            image_bytes = base64.b64decode(data_b64)
        except (ValueError, TypeError):
            return
        self.on_camera_image(image_bytes)
