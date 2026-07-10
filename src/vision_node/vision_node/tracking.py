import math

# keypoint를 신뢰할 최소 confidence - 이 미만이면 없는 것으로 보고 bbox 중심 폴백.
# YOLO pose의 kpt conf는 "그 점이 실제로 보이는가"에 가까워 0.5면 보수적으로 안전.
KPT_CONF_MIN = 0.5


def detection_anchor(det):
    """검출의 추적 기준 픽셀과 그 출처(mode)를 돌려준다.

    mode:
      'mid'      두 kpt 모두 유효 - keypoint 중점(=파지점, 중심 파지 계약)
      'p0'/'p1'  한쪽 kpt만 유효(근접 시 반대쪽 끝이 화면 밖으로 잘리면 모델이
                 그 kpt conf를 0 근처로 떨어뜨린다 - 라벨 스펙 v0 학습 결과) -
                 호출측(ToolTracker)이 학습해 둔 3D 오프셋으로 파지점을 추정한다
      'bbox'     kpt 없음(구 box 모델)/양쪽 저신뢰 - bbox 중심

    저신뢰 kpt의 xy는 실좌표가 오염돼 있어(라이브런에서 kpt0에 2px까지 붙는 퇴화
    관측) 절대 그대로 쓰면 안 된다 - conf 문턱을 낮추는 게 아니라 버리는 게 맞다."""
    c0 = getattr(det, 'kpt0_conf', 0.0)
    c1 = getattr(det, 'kpt1_conf', 0.0)
    if c0 >= KPT_CONF_MIN and c1 >= KPT_CONF_MIN:
        return (det.kpt0_x + det.kpt1_x) / 2.0, (det.kpt0_y + det.kpt1_y) / 2.0, 'mid'
    if c0 >= KPT_CONF_MIN:
        return det.kpt0_x, det.kpt0_y, 'p0'
    if c1 >= KPT_CONF_MIN:
        return det.kpt1_x, det.kpt1_y, 'p1'
    return (det.x1 + det.x2) / 2.0, (det.y1 + det.y2) / 2.0, 'bbox'


def detection_center(det):
    """(표시/로깅용 하위호환) 기준 픽셀 좌표만. 단일 kpt 모드는 오프셋 추정 없이는
    파지점이 아니므로 여기서는 bbox 중심으로 취급한다 - 추적 본체는 detection_anchor
    + ToolTracker의 오프셋 보정을 쓴다."""
    cx, cy, mode = detection_anchor(det)
    if mode in ('p0', 'p1'):
        return (det.x1 + det.x2) / 2.0, (det.y1 + det.y2) / 2.0
    return cx, cy


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

    # kpt->파지점 오프셋 EMA 계수 (새 관측 가중치). 도구가 정지 상태라 오프셋이
    # 거의 불변이므로 낮게 잡아 노이즈를 누른다.
    OFFSET_EMA_ALPHA = 0.3

    def __init__(self, alpha=0.6, beta=0.3, alpha_z=None):
        # x,y는 검출 좌표 자체가 이미 안정적이라 EMA를 적용하지 않는다(raw 그대로
        # 사용, 2026-07-10) - 스무딩하면 접근 중 지연만 생긴다. alpha는 이제 xy에는
        # 안 쓰이고 alpha_z 기본값 산출(미지정 시 하위 호환)에만 남는다.
        self.alpha = alpha
        self.beta = beta     # 속도 스무딩 계수
        # z(depth) 전용 스무딩 계수. RealSense 스테레오 depth는 정지 물체에서도
        # 프레임간 temporal jitter가 본질적으로 있어(패치 median은 "한 프레임 내" 공간
        # 이상치만 걸러줄 뿐 이 프레임간 흔들림엔 무력하다) z만 EMA로 누른다.
        # 지정 없으면 alpha와 동일(과거 동작과 호환).
        self.alpha_z = alpha if alpha_z is None else alpha_z
        self.position = None       # 마지막 추정 위치 (x, y, z) base_link
        self.velocity = (0.0, 0.0)
        self.last_valid_z = None   # depth_valid=False일 때 유지할 마지막 z
        self.last_time = None
        # "p0 -> 파지점" 3D 오프셋(base). 두 kpt가 다 보이는 프레임에서 학습해 두고,
        # 접근 중 한쪽 끝이 화면 밖으로 잘리면(단일 kpt 모드) 남은 kpt + 이 오프셋으로
        # 파지점을 유지한다 - 잘린 bbox 중심 폴백의 위쪽 편향(라이브런 실측)을 막는다.
        self.kpt_offset = None

    def reset(self):
        """set_mode로 TRACK_TOOL에 새로 진입할 때 호출 - 이전 추적 상태를 지운다."""
        self.position = None
        self.velocity = (0.0, 0.0)
        self.last_valid_z = None
        self.last_time = None
        self.kpt_offset = None

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

        # 2. 각 후보의 기준점(keypoint 중점=파지점)을 3D로 복원. 한쪽 kpt만 유효한
        #    후보는 학습된 오프셋으로 파지점을 추정하고, 오프셋 이력이 없으면(이 도구를
        #    두 kpt로 본 적이 없음) 기존 bbox 중심 폴백을 유지한다.
        reconstructed = []
        for d in candidates:
            cx, cy, mode = detection_anchor(d)
            if mode in ('p0', 'p1') and self.kpt_offset is None:
                cx, cy = (d.x1 + d.x2) / 2.0, (d.y1 + d.y2) / 2.0
                mode = 'bbox'
            r = reconstruct_fn(cx, cy, d.x2 - d.x1, d.y2 - d.y1)
            if r is None:
                continue
            x, y, z, dv = r
            if mode == 'p0':
                x += self.kpt_offset[0]
                y += self.kpt_offset[1]
                z += self.kpt_offset[2]
            elif mode == 'p1':
                # 파지점은 정확히 두 끝점의 중점이라 p1 기준 오프셋은 부호 반전과 같다
                x -= self.kpt_offset[0]
                y -= self.kpt_offset[1]
                z -= self.kpt_offset[2]
            reconstructed.append(((x, y, z, dv), d.score, d, mode))
        if not reconstructed:
            return None

        # 3. 후보 선택: 이전 추정이 없으면(첫 프레임) 최고 score, 있으면 최근접 매칭
        if self.position is None:
            chosen, _, chosen_det, chosen_mode = max(reconstructed, key=lambda item: item[1])
        else:
            def dist(item):
                r = item[0]
                return math.dist((r[0], r[1], r[2]), self.position)
            chosen, _, chosen_det, chosen_mode = min(reconstructed, key=dist)
        x, y, z, depth_valid = chosen

        # 3.5. 두 kpt가 다 보이는 동안 "p0 -> 파지점" 오프셋(base)을 학습해 둔다 -
        # 접근 중 한쪽 끝이 잘리는 순간을 대비한 상태 준비 (위 2번에서 소비).
        if chosen_mode == 'mid' and depth_valid:
            r0 = reconstruct_fn(chosen_det.kpt0_x, chosen_det.kpt0_y,
                                 chosen_det.x2 - chosen_det.x1,
                                 chosen_det.y2 - chosen_det.y1)
            if r0 is not None and r0[3]:
                off = (x - r0[0], y - r0[1], z - r0[2])
                if self.kpt_offset is None:
                    self.kpt_offset = off
                else:
                    a = self.OFFSET_EMA_ALPHA
                    self.kpt_offset = tuple(
                        a * n + (1.0 - a) * o for n, o in zip(off, self.kpt_offset))

        # 4. depth 무효 구간: z는 마지막 유효값으로 고정(전체 계획.md 2.7절)
        if depth_valid:
            self.last_valid_z = z
        elif self.last_valid_z is not None:
            z = self.last_valid_z

        position, velocity, depth_valid = self._filter_update(x, y, z, depth_valid, stamp)
        return position, velocity, depth_valid, chosen_det

    def _filter_update(self, x, y, z, depth_valid, stamp):
        """알파-베타 필터 한 스텝: x,y는 raw 그대로, z는 alpha_z로, 속도는 beta로 스무딩."""
        if self.position is None or self.last_time is None:
            # 첫 프레임은 스무딩할 이전 값이 없으니 그대로 채택, 속도는 0
            self.position = (x, y, z)
            self.velocity = (0.0, 0.0)
            self.last_time = stamp
            return self.position, self.velocity, depth_valid

        dt = max(stamp - self.last_time, 1e-3)
        raw_vx = (x - self.position[0]) / dt
        raw_vy = (y - self.position[1]) / dt

        smoothed_x = x
        smoothed_y = y
        smoothed_z = self.position[2] + self.alpha_z * (z - self.position[2])
        vx = self.velocity[0] + self.beta * (raw_vx - self.velocity[0])
        vy = self.velocity[1] + self.beta * (raw_vy - self.velocity[1])

        self.position = (smoothed_x, smoothed_y, smoothed_z)
        self.velocity = (vx, vy)
        self.last_time = stamp
        return self.position, self.velocity, depth_valid
