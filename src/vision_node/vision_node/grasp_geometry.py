"""파지 기하 계산 유틸

bbox+뎁스에서 파지에 필요한 정보(패치 median 뎁스, 장축/그립 yaw, 베이스 좌표 변환)를
계산한다. tracking.py와 같은 이유로 ROS 타입에 의존하지 않는다.
"""
import cv2
import numpy as np


def align_depth_to_color(
        depth_m, depth_fx, depth_fy, depth_ppx, depth_ppy,
        color_fx, color_fy, color_ppx, color_ppy,
        rotation, translation, out_shape, dmin=0.10, dmax=2.0):
    """뎁스 이미지(뎁스 광학 좌표계, meters)를 컬러 픽셀 격자에 정렬한다.

    realsense2_camera의 enable_sync+align_depth가 이 카메라/드라이버 조합(FW 5.13.0.50,
    ROS wrapper v4.57.7)에서 근본적으로 깨져 있어(2026-07-12 확인 - 켜면 depth 프레임이
    요청 fps의 정확히 2배로 나오고 aligned_depth_to_color가 0프레임만 발행, 해상도/fps를
    바꿔도 재현) 드라이버 정렬에 의존하지 않고 직접 계산한다. 카메라 하드웨어/뎁스 센서
    자체는 정상임을 rs-hello-realsense(SDK 직접 호출)로 별도 확인함.

    cv2.rgbd.registerDepth(OpenCV contrib rgbd 모듈)로 계산한다 - 뎁스 클라우드 생성 ->
    (rotation, translation)으로 컬러 광학 좌표계 변환 -> 컬러 intrinsics로 재투영이라는
    동일 알고리즘을 최적화된 C++로 제공한다. 처음엔 numpy(np.minimum.at 기반 z-buffer
    스캐터)로 직접 구현했으나 424x240 기준 프레임당 ~10-12ms가 걸려(np.minimum.at이 그중
    절반 이상) "카메라가 짧게 끊기고 로봇팔 접근이 느려짐"의 원인이 됐다(2026-07-12 실기
    확인) - cv2.rgbd.registerDepth는 동일 조건에서 ~2.8ms/frame으로 약 4배 빠르고, 기존
    numpy 구현의 유닛 테스트(항등변환/이동/occlusion z-buffer/전체 무효) 시나리오를 그대로
    통과시켜 결과가 float32 정밀도(~1e-7) 이내로 동일함을 확인했다.

    주의: cv2.rgbd.registerDepth는 "negative depth 값을 체크하지 않는다"(자체 문서)뿐
    아니라 **0도 체크하지 않는다** - 우리 코드 전체의 "0=무효" 관례를 모른다. 0을 그대로
    넣으면 그 픽셀이 z=0인 실제 점처럼 처리되다 재투영식의 1/z에서 NaN이 나고 그 NaN이
    주변으로 번진다(실측 확인: 항등변환 테스트에서 배경 0이 그대로면 유효 영역까지 NaN으로
    오염됨). 그래서 dmin/dmax로 무효 판정한 픽셀을 미리 NaN으로 표시해 넣어야
    correct하게 무효로만 전파되고(실측 확인: occlusion 테스트에서 더 가까운 점만 정확히
    남음), 출력의 NaN을 다시 0.0으로 되돌려 기존 반환 계약(0=무효, meters, float64)을
    그대로 지킨다.

    rotation: 3x3 배열형(list 또는 ndarray), translation: 길이 3 배열형 - 둘 다
    tracking.transform_to_matrix/quaternion_to_rotation_matrix와 동일하게 tf_buffer의
    TransformStamped(뎁스->컬러)에서 만든다.
    """
    out_h, out_w = out_shape
    depth_in = depth_m.astype(np.float32).copy()
    invalid = ~((depth_in > dmin) & (depth_in < dmax))
    depth_in[invalid] = np.nan
    k_depth = np.array(
        [[depth_fx, 0.0, depth_ppx], [0.0, depth_fy, depth_ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64)
    k_color = np.array(
        [[color_fx, 0.0, color_ppx], [0.0, color_fy, color_ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64)
    dist_coeffs = np.zeros(5, dtype=np.float64)  # RealSense 컬러 이미지는 이미 rectify됨
    rt = np.eye(4, dtype=np.float64)
    rt[:3, :3] = np.asarray(rotation, dtype=np.float64)
    rt[:3, 3] = np.asarray(translation, dtype=np.float64)
    registered = cv2.rgbd.registerDepth(
        k_depth, k_color, dist_coeffs, rt, depth_in, (out_w, out_h), depthDilation=False)
    return np.nan_to_num(registered, nan=0.0).astype(np.float64)


def patch_median_depth(depth_m, cx, cy, half=4, dmin=0.10, dmax=2.0):
    """(cx, cy) 주변 (2*half+1)^2 패치의 유효 뎁스 median(m)과 유효 픽셀 비율을 반환.

    단일 픽셀은 금속/반사면 뎁스 구멍에 취약해서 패치로 보완한다.
    유효 픽셀이 하나도 없으면 (None, 0.0).
    """
    h, w = depth_m.shape
    px = min(max(int(cx), 0), w - 1)
    py = min(max(int(cy), 0), h - 1)
    patch = depth_m[max(py - half, 0):py + half + 1, max(px - half, 0):px + half + 1]
    valid = patch[(patch > dmin) & (patch < dmax)]
    ratio = valid.size / patch.size if patch.size else 0.0
    if valid.size == 0:
        return None, ratio
    return float(np.median(valid)), ratio


def tool_axis_from_depth(roi_depth_m, fx, fy, ppx, ppy, ox=0, oy=0,
                          dmin=0.10, dmax=2.0, band_m=0.008, min_px=50):
    """bbox ROI 뎁스에서 공구 윗면 마스크를 만들어 3D 포인트클라우드 PCA로 장축을 구한다.

    공구 높이가 ~1cm라 중심 뎁스 +-대칭 밴드로는 벨트가 섞여 들어옴 ->
    근거리 percentile(p10)을 윗면 깊이로 잡고 그보다 band_m 이상 깊은 픽셀(벨트)은 제외.
    스펙클 제거 후 가장 큰 덩어리만 쓴다.

    각도는 마스크 픽셀을 (fx,fy,ppx,ppy)로 실제 3D 좌표(m)로 역투영한 뒤 공분산
    고유분해로 구한다. 컨베이어 벨트 위 공구는 한쪽이 벨트 가장자리 밖 허공에 걸쳐
    기울어진 채로 놓이는 일이 흔한데, 이 경우 XY 실루엣만으로는 장단축이 애매해도
    (예: 렌치/망치 머리처럼 폭이 넓은 부분) 뎁스 방향 기울기가 실제 축을 알려준다.
    완전히 평평하게 누운 경우엔 Z 분산이 노이즈 수준이라 사실상 XY가 지배적이므로
    기존 2D 모멘트 방식과 결과가 수렴한다.

    ox, oy: ROI가 원본 프레임에서 잘려나온 좌상단 오프셋(픽셀) - 역투영 시 필요.

    반환: (장축 각도 deg [0,180), 시각화용 minAreaRect(2D), 장단축비) 또는
    실패 시 (None, None, None). 장단축비는 최대/2번째 고유값의 제곱근 비율로,
    1에 가까울수록(정사각형에 가까운 덩어리) 각도가 노이즈에 민감하다는 뜻이다.
    """
    roi_valid = roi_depth_m[(roi_depth_m > dmin) & (roi_depth_m < dmax)]
    if roi_valid.size < min_px:
        return None, None, None
    z_top = float(np.percentile(roi_valid, 10))  # ROI 안 가장 가까운 면 = 공구 윗면
    mask = ((roi_depth_m > z_top - 0.005) & (roi_depth_m < z_top + band_m)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if cv2.contourArea(c) >= min_px]
    if not candidates:
        return None, None, None
    blob = max(candidates, key=cv2.contourArea)
    rect = cv2.minAreaRect(blob)  # 시각화용 - 화면 좌표계라 2D로 그대로 유지

    blob_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(blob_mask, [blob], -1, 1, thickness=-1)
    vs, us = np.nonzero(blob_mask)
    zs = roi_depth_m[vs, us]
    xs = (us + ox - ppx) * zs / fx
    ys = (vs + oy - ppy) * zs / fy
    pts = np.stack([xs, ys, zs], axis=1)

    pts_c = pts - pts.mean(axis=0)
    cov = pts_c.T @ pts_c / len(pts_c)
    eigvals, eigvecs = np.linalg.eigh(cov)  # 오름차순 정렬
    principal = eigvecs[:, -1]
    axis_deg = np.degrees(np.arctan2(principal[1], principal[0])) % 180
    elongation = float(np.sqrt(max(eigvals[-1], 0.0) / max(eigvals[-2], 1e-12)))
    return axis_deg, rect, elongation


class AxisSmoother:
    """장축 각도의 시간적 스무딩. 각도는 180도 주기라 2배각 단위벡터로 EMA 한다
    (0<->179도 경계에서 튀지 않음). 키(클래스 id 등)별로 상태를 유지한다."""

    def __init__(self, alpha=0.25):
        self.alpha = alpha  # 1=스무딩 없음, 작을수록 부드럽지만 반응 느림
        self._state = {}

    def update(self, key, axis_deg, alpha=None):
        """새 관측을 반영한 스무딩된 각도(deg)를 반환. alpha를 지정하면 이번 호출만
        기본값 대신 그 값을 쓴다 - 관측 신뢰도가 낮을 때(예: 덩어리가 정사각형에 가까워
        PCA 각도가 불안정할 때) 호출 측에서 더 강하게 누르는 용도."""
        a = self.alpha if alpha is None else alpha
        th2 = np.deg2rad(2.0 * axis_deg)
        vec = np.array([np.cos(th2), np.sin(th2)])
        prev = self._state.get(key)
        if prev is not None:
            vec = a * vec + (1.0 - a) * prev
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
        self._state[key] = vec
        return (np.degrees(np.arctan2(vec[1], vec[0])) / 2.0) % 180

    def current(self, key):
        """마지막 스무딩 상태의 각도(deg) 또는 이력이 없으면 None. 상태는 갱신하지
        않는다 - 관측이 불가능한 프레임(근접 시 keypoint 한쪽이 화면 밖으로 잘려
        벡터각을 못 구할 때)에 직전 축을 유지(hold)하는 용도."""
        vec = self._state.get(key)
        if vec is None:
            return None
        return float((np.degrees(np.arctan2(vec[1], vec[0])) / 2.0) % 180)

    def reset(self, key):
        """물체가 화면 밖으로 잘리는 등 이력이 오염될 상황에서 호출 - 다음 관측부터 새로 시작."""
        self._state.pop(key, None)

    def reset_missing(self, seen_keys):
        """이번 프레임에 보이지 않는 클래스의 이력을 지운다 - 물체가 화면에서 완전히
        사라졌다가 다시 나타났을 때 이전 물체의 각도가 남아 새 물체 각도와 섞이는 것을 막는다."""
        for key in list(self._state.keys()):
            if key not in seen_keys:
                self._state.pop(key, None)


def is_bbox_at_edge(x1, y1, x2, y2, width, height, margin_px=8):
    """bbox가 화면 가장자리에 닿았는지 - 닿았으면 물체가 잘려 보이는 상태라 yaw를 신뢰할 수 없다."""
    return (x1 <= margin_px or y1 <= margin_px
            or x2 >= width - margin_px or y2 >= height - margin_px)


def zyz_deg_to_rot(a_deg, b_deg, c_deg):
    """Doosan posx의 ZYZ 오일러 각(deg) -> 3x3 회전행렬. R = Rz(A) @ Ry(B) @ Rz(C)."""
    a, b, c = np.deg2rad([a_deg, b_deg, c_deg])
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rz_a = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    ry_b = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz_c = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rz_a @ ry_b @ rz_c


def posx_to_matrix(posx):
    """Doosan posx [x,y,z,A,B,C] -> 4x4 변환행렬 (base -> gripper/TCP, 단위 mm)."""
    T = np.eye(4)
    T[:3, :3] = zyz_deg_to_rot(posx[3], posx[4], posx[5])
    T[:3, 3] = posx[:3]
    return T


def yaw_deg_to_quaternion(yaw_deg):
    """Z축 회전 yaw(deg) -> 쿼터니언 (x, y, z, w)."""
    half = np.deg2rad(yaw_deg) / 2.0
    return (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))
