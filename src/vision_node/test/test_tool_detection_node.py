import numpy as np
import pytest
import rclpy
from std_msgs.msg import Header
from sensor_msgs.msg import Image

from vision_node.tool_detection_node import ToolDetectionNode


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = ToolDetectionNode()
    yield n
    n.destroy_node()


def _make_image_msg():
    msg = Image()
    msg.header = Header()
    msg.header.frame_id = 'camera_color_optical_frame'
    return msg


class _FakeResult:
    boxes = []
    keypoints = None


def test_on_color_survives_model_predict_exception(node, monkeypatch):
    """YOLO 추론(model.predict)이 예외를 던져도 노드가 죽지 않고 이번 프레임만 건너뛰어야
    한다 - 기본 rclpy.spin()은 SingleThreadedExecutor를 쓰는데, 콜백에서 새어나온 예외를
    스핀 루프가 그대로 재발생시켜 프로세스를 종료시킨다. respawn을 걸어도 죽는 순간
    /detection/tool_boxes가 끊기고, vision_node의 4토픽 동기화 콜백이 다시는 안 돌아
    디버그 영상까지 함께 영구히 멈춘다(2026-07-13, "태스크 지시 순간 카메라가 죽는다"
    조사 중 발견한 별개의 확정 결함)."""
    monkeypatch.setattr(
        node._bridge, 'imgmsg_to_cv2',
        lambda msg, desired_encoding=None: np.zeros((240, 424, 3), dtype=np.uint8))

    def _boom(*args, **kwargs):
        raise RuntimeError('inference boom')

    monkeypatch.setattr(node.model, 'predict', _boom)

    published = []
    node.pub_detections.publish = published.append

    node._on_color(_make_image_msg())  # 여기서 예외가 새어나오면 테스트 자체가 실패한다

    assert published == []


def test_on_color_recovers_after_exception_on_next_frame(node, monkeypatch):
    """예외가 난 프레임 다음부터는 다시 정상 발행돼야 한다 - 상태가 반쯤 갱신된 채로
    다음 호출에 새지 않는지 확인."""
    monkeypatch.setattr(
        node._bridge, 'imgmsg_to_cv2',
        lambda msg, desired_encoding=None: np.zeros((240, 424, 3), dtype=np.uint8))

    call_count = [0]

    def _predict(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError('inference boom')
        return [_FakeResult()]

    monkeypatch.setattr(node.model, 'predict', _predict)

    published = []
    node.pub_detections.publish = published.append

    node._on_color(_make_image_msg())  # 1번째: 예외 - 건너뜀
    node._on_color(_make_image_msg())  # 2번째: 정상 - 발행

    assert len(published) == 1
