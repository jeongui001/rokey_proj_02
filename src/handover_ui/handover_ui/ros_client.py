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
            'HANDOVER_UI_CAMERA_TOPIC', DEFAULT_CAMERA_TOPIC)
        self.reconnect_interval_s = reconnect_interval_s

        self.on_task_status = None       # (state, detail, operation_mode, safety_state)
        self.on_gripper_state = None      # (width_mm, grip_detected)
        self.on_fault = None              # (message)
        self.on_camera_image = None       # (image_bytes)
        self.on_connection_changed = None  # (is_connected: bool)

        self.ros = roslibpy.Ros(host=host, port=port)
        self._command_topic = None
        self._subscribed_topics = []  # 구독 중인 roslibpy.Topic 목록 (중복 구독 방지용)
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
        """연결이 끊겨 있으면 재연결을 시도한다. UI 쪽 타이머가 주기적으로 호출한다."""
        if self._closed:
            return
        if self.is_connected():
            return
        self.connect()

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
        self.subscribe_all()
        if self.on_connection_changed is not None:
            self.on_connection_changed(True)

    def _on_close(self, *_args):
        # 연결이 끊겼으므로 기존 구독은 더 이상 유효하지 않다 - 정리해서, 재연결
        # 성공(on_ready) 시 subscribe_all이 토픽마다 정확히 한 번씩 다시 구독하게 한다.
        self._teardown_subscriptions()
        if self.on_connection_changed is not None:
            self.on_connection_changed(False)

    def _on_error(self, *_args):
        if self.on_connection_changed is not None:
            self.on_connection_changed(False)

    # ---- 구독/발행 ----

    def subscribe_all(self):
        """필요한 토픽을 구독한다. 이미 구독 중이면(on_ready가 여러 번 호출돼도)
        아무 것도 하지 않는다 - 실제 재구독은 _on_close에서 목록을 비운 뒤,
        재연결 성공 시 이 메서드가 다시 호출될 때 정확히 한 번씩 일어난다."""
        with self._subscribe_lock:
            if self._subscribed_topics:
                return
            specs = (
                ('/task/status', 'std_msgs/String', self._on_task_status_raw),
                ('/gripper/state', 'handover_interfaces/GripperState', self._on_gripper_state_raw),
                ('/robot/fault', 'std_msgs/String', self._on_fault_raw),
                (self.camera_topic, 'sensor_msgs/CompressedImage', self._on_camera_image_raw),
            )
            for name, msg_type, callback in specs:
                topic = roslibpy.Topic(self.ros, name, msg_type)
                topic.subscribe(callback)
                self._subscribed_topics.append(topic)
            if self._command_topic is None:
                self._command_topic = roslibpy.Topic(
                    self.ros, '/user_command/text', 'std_msgs/String')

    def _teardown_subscriptions(self):
        """구독 중인 토픽을 unsubscribe하고, command 토픽은 unadvertise한 뒤 내부
        목록을 정리한다. 개별 실패가 있어도 UI(호출측)가 죽지 않도록 예외를 삼킨다."""
        with self._subscribe_lock:
            topics = self._subscribed_topics
            self._subscribed_topics = []
            command_topic = self._command_topic
            self._command_topic = None
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
