import math


def pixel_to_camera_xyz(px, py, depth, fx, fy, ppx, ppy):
    """픽셀 좌표 + depth(camera 기준 z)를 camera 좌표계 3D 점으로 변환."""
    x = (px - ppx) * depth / fx
    y = (py - ppy) * depth / fy
    return x, y, depth


def quaternion_to_rotation_matrix(x, y, z, w):
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def transform_to_matrix(translation, rotation):
    """translation=(x,y,z), rotation=(x,y,z,w) 쿼터니언 -> 4x4 변환행렬(중첩 리스트)."""
    r = quaternion_to_rotation_matrix(*rotation)
    return [
        [r[0][0], r[0][1], r[0][2], translation[0]],
        [r[1][0], r[1][1], r[1][2], translation[1]],
        [r[2][0], r[2][1], r[2][2], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def camera_to_base(camera_xyz, tf_matrix):
    x, y, z = camera_xyz
    out = []
    for row in tf_matrix[:3]:
        out.append(row[0] * x + row[1] * y + row[2] * z + row[3])
    return tuple(out)


def is_approaching(position_xy, velocity_xy, ref_xy):
    dx = ref_xy[0] - position_xy[0]
    dy = ref_xy[1] - position_xy[1]
    dot = velocity_xy[0] * dx + velocity_xy[1] * dy
    return dot > 0.0


class ToolTracker:
    """vision_node의 TRACK_TOOL 단순 추적기: 최근접 매칭 + 알파-베타 속도 필터."""

    def __init__(self, alpha=0.6, beta=0.3):
        self.alpha = alpha
        self.beta = beta
        self.position = None
        self.velocity = (0.0, 0.0)
        self.last_valid_z = None
        self.last_time = None

    def reset(self):
        self.position = None
        self.velocity = (0.0, 0.0)
        self.last_valid_z = None
        self.last_time = None

    def update(self, detections, tool_class, reconstruct_fn, stamp):
        candidates = [d for d in detections if d.class_name == tool_class]
        if not candidates:
            return None

        reconstructed = []
        for d in candidates:
            cx = (d.x1 + d.x2) / 2.0
            cy = (d.y1 + d.y2) / 2.0
            r = reconstruct_fn(cx, cy)
            if r is not None:
                reconstructed.append((r, d.score))
        if not reconstructed:
            return None

        if self.position is None:
            chosen, _ = max(reconstructed, key=lambda item: item[1])
        else:
            def dist(item):
                r = item[0]
                return math.dist((r[0], r[1], r[2]), self.position)
            chosen, _ = min(reconstructed, key=dist)

        x, y, z, depth_valid = chosen
        if depth_valid:
            self.last_valid_z = z
        elif self.last_valid_z is not None:
            z = self.last_valid_z

        return self._filter_update(x, y, z, depth_valid, stamp)

    def _filter_update(self, x, y, z, depth_valid, stamp):
        if self.position is None or self.last_time is None:
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
