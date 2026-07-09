import math


def pixel_to_camera_xyz(px, py, depth, fx, fy, ppx, ppy):
    """픽셀 좌표 + depth(camera 기준 z)를 camera 좌표계 3D 점으로 변환 (핀홀 카메라 역투영)."""
    x = (px - ppx) * depth / fx
    y = (py - ppy) * depth / fy
    return x, y, depth


def quaternion_to_rotation_matrix(x, y, z, w):
    """쿼터니언(x,y,z,w) -> 3x3 회전행렬. tf의 rotation을 행렬 연산에 쓰기 위한 변환."""
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def transform_to_matrix(translation, rotation):
    """translation=(x,y,z), rotation=(x,y,z,w) 쿼터니언 -> 4x4 변환행렬(중첩 리스트).
    tf_buffer.lookup_transform()이 돌려주는 TransformStamped를 이 형태로 바꿔서
    camera_to_base()에 넘긴다."""
    r = quaternion_to_rotation_matrix(*rotation)
    return [
        [r[0][0], r[0][1], r[0][2], translation[0]],
        [r[1][0], r[1][1], r[1][2], translation[1]],
        [r[2][0], r[2][1], r[2][2], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def camera_to_base(camera_xyz, tf_matrix):
    """camera 좌표계의 점을 tf_matrix(base<-camera)로 base_link 좌표로 변환."""
    x, y, z = camera_xyz
    out = []
    for row in tf_matrix[:3]:
        out.append(row[0] * x + row[1] * y + row[2] * z + row[3])
    return tuple(out)


def is_approaching(position_xy, velocity_xy, ref_xy):
    """속도 벡터가 기준점(ref_xy, 파지 구역) 쪽을 향하고 있으면 True.
    (기준점 - 현재위치) 방향과 속도 방향의 내적 부호로 판정 - 내적이 양수면 같은 방향."""
    dx = ref_xy[0] - position_xy[0]
    dy = ref_xy[1] - position_xy[1]
    dot = velocity_xy[0] * dx + velocity_xy[1] * dy
    return dot > 0.0


class ToolTracker:
    """vision_node의 TRACK_TOOL 단순 추적기: 최근접 매칭 + 알파-베타 속도 필터.

    이 필터는 ToolTrack.velocity(표시·approaching 판정용) 계산 전용이다.
    실제 서보 제어에 쓰이는 정밀 칼만 필터는 robot_control/kalman.py에 별도로 있다.
    """

    def __init__(self, alpha=0.6, beta=0.3):
        self.alpha = alpha   # 위치 스무딩 계수
        self.beta = beta     # 속도 스무딩 계수
        self.position = None       # 마지막 추정 위치 (x, y, z) base_link
        self.velocity = (0.0, 0.0)
        self.last_valid_z = None   # depth_valid=False일 때 유지할 마지막 z
        self.last_time = None

    def reset(self):
        """set_mode로 TRACK_TOOL에 새로 진입할 때 호출 - 이전 추적 상태를 지운다."""
        self.position = None
        self.velocity = (0.0, 0.0)
        self.last_valid_z = None
        self.last_time = None

    def update(self, detections, tool_class, reconstruct_fn, stamp):
        """한 프레임의 검출 목록(detections)을 받아 추적 상태를 한 스텝 갱신한다.

        reconstruct_fn(cx, cy, bbox_w, bbox_h) -> (x, y, z, depth_valid) 또는 None:
        bbox 중심 픽셀을 3D로 복원하는 함수. bbox_w/bbox_h는 depth 패치 크기가
        bbox 밖(배경)으로 새지 않게 제한하는 데 쓴다. vision_node.py가 depth
        이미지·intrinsics·tf를 클로저로 캡처해서 넘겨준다(이 파일은 ROS 타입을
        몰라도 되게 하기 위한 분리).

        반환: (position, velocity, depth_valid, chosen_det) 또는 이번 프레임에
        tool_class와 일치하는 검출이 하나도 없으면 None. chosen_det은 선택된 원본
        검출(Detection2D) - 호출측이 그 bbox로 depth ROI를 잘라 장축(yaw)을 계산하는
        데 쓴다.
        """
        # 1. 원하는 클래스의 검출만 후보로 추림
        candidates = [d for d in detections if d.class_name == tool_class]
        if not candidates:
            return None

        # 2. 각 후보의 bbox 중심을 3D로 복원
        reconstructed = []
        for d in candidates:
            cx = (d.x1 + d.x2) / 2.0
            cy = (d.y1 + d.y2) / 2.0
            r = reconstruct_fn(cx, cy, d.x2 - d.x1, d.y2 - d.y1)
            if r is not None:
                reconstructed.append((r, d.score, d))
        if not reconstructed:
            return None

        # 3. 후보 선택: 이전 추정이 없으면(첫 프레임) 최고 score, 있으면 최근접 매칭
        if self.position is None:
            chosen, _, chosen_det = max(reconstructed, key=lambda item: item[1])
        else:
            def dist(item):
                r = item[0]
                return math.dist((r[0], r[1], r[2]), self.position)
            chosen, _, chosen_det = min(reconstructed, key=dist)

        # 4. depth 무효 구간: z는 마지막 유효값으로 고정(전체 계획.md 2.7절)
        x, y, z, depth_valid = chosen
        if depth_valid:
            self.last_valid_z = z
        elif self.last_valid_z is not None:
            z = self.last_valid_z

        position, velocity, depth_valid = self._filter_update(x, y, z, depth_valid, stamp)
        return position, velocity, depth_valid, chosen_det

    def _filter_update(self, x, y, z, depth_valid, stamp):
        """알파-베타 필터 한 스텝: 위치는 alpha로, 속도는 beta로 각각 스무딩."""
        if self.position is None or self.last_time is None:
            # 첫 프레임은 스무딩할 이전 값이 없으니 그대로 채택, 속도는 0
            self.position = (x, y, z)
            self.velocity = (0.0, 0.0)
            self.last_time = stamp
            return self.position, self.velocity, depth_valid

        dt = max(stamp - self.last_time, 1e-3)
        raw_vx = (x - self.position[0]) / dt
        raw_vy = (y - self.position[1]) / dt

        smoothed_x = self.position[0] + self.alpha * (x - self.position[0])
        smoothed_y = self.position[1] + self.alpha * (y - self.position[1])
        vx = self.velocity[0] + self.beta * (raw_vx - self.velocity[0])
        vy = self.velocity[1] + self.beta * (raw_vy - self.velocity[1])

        self.position = (smoothed_x, smoothed_y, z)
        self.velocity = (vx, vy)
        self.last_time = stamp
        return self.position, self.velocity, depth_valid
