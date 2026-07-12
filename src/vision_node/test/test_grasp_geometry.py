"""grasp_geometry 순수 함수 검증 - 합성 뎁스 이미지로 하드웨어 없이 계산 로직을 확인한다."""
import numpy as np
import pytest

from vision_node.grasp_geometry import (
    AxisSmoother, align_depth_to_color, is_bbox_at_edge, patch_median_depth,
    posx_to_matrix, tool_axis_from_depth, yaw_deg_to_quaternion, zyz_deg_to_rot,
)


FAKE_INTR = dict(fx=600.0, fy=600.0, ppx=100.0, ppy=100.0)  # 테스트용 가짜 intrinsics


def _synthetic_roi(angle_deg, size=200, belt_z=0.31, tool_z=0.30):
    """벨트(belt_z) 위에 angle_deg로 놓인 막대(tool_z) 모양의 뎁스 ROI."""
    import cv2
    roi = np.full((size, size), belt_z, dtype=np.float64)
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    box = cv2.boxPoints((center, (140, 30), angle_deg))
    cv2.fillPoly(mask, [box.astype(np.int32)], 1)
    roi[mask == 1] = tool_z
    return roi


def _synthetic_tilted_roi(angle_deg, size=120, belt_z=0.31, z_near=0.296, z_far=0.302,
                           box_size=(60, 55)):
    """실루엣은 거의 정사각형(장단축비 낮음)이지만 장축 방향으로 뎁스가 선형으로 기우는
    ROI - 벨트 가장자리 밖 허공에 걸쳐 기운 물체를 흉내낸다. 2D 실루엣만으로는 각도가
    애매해도 뎁스 기울기가 실제 축을 알려줘야 한다."""
    import cv2
    roi = np.full((size, size), belt_z, dtype=np.float64)
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    box = cv2.boxPoints((center, box_size, angle_deg))
    cv2.fillPoly(mask, [box.astype(np.int32)], 1)
    theta = np.deg2rad(angle_deg)
    ux, uy = np.cos(theta), np.sin(theta)
    ys_idx, xs_idx = np.nonzero(mask)
    proj = (xs_idx - center[0]) * ux + (ys_idx - center[1]) * uy
    proj_norm = (proj - proj.min()) / max(proj.max() - proj.min(), 1e-6)
    roi[ys_idx, xs_idx] = z_near + proj_norm * (z_far - z_near)
    return roi


def test_align_depth_identity_transform_reproduces_input():
    # 회전 없음 + 이동 없음 + 동일 intrinsics면 각 픽셀이 정확히 자기 자신에게
    # 재투영돼야 한다(핀홀 역투영 후 재투영이 항등연산으로 상쇄됨) - realsense 드라이버의
    # align_depth를 대체하는 이 함수의 가장 기본적인 정합성 검증. 내부적으로
    # cv2.rgbd.registerDepth(float32)를 거치므로 1e-9는 너무 빡빡함 - 실측 오차는
    # ~5e-8 수준이라 1e-6이면 충분히 여유 있게 float32 라운딩만 허용한다.
    depth = np.zeros((50, 60), dtype=np.float64)
    depth[10:20, 15:25] = 0.5
    depth[30:35, 40:45] = 1.2
    aligned = align_depth_to_color(
        depth, depth_fx=600.0, depth_fy=600.0, depth_ppx=30.0, depth_ppy=25.0,
        color_fx=600.0, color_fy=600.0, color_ppx=30.0, color_ppy=25.0,
        rotation=np.eye(3), translation=np.zeros(3), out_shape=(50, 60))
    assert aligned == pytest.approx(depth, abs=1e-6)


def test_align_depth_translation_shifts_point():
    # 뎁스 카메라가 컬러 카메라보다 +x 방향으로 치우쳐 있으면(뎁스->컬러 변환의 translation.x
    # 가 양수) 재투영된 점은 원래보다 왼쪽(작은 u)으로 이동한다 - 핀홀 재투영 u = x*fx/z+ppx에서
    # x가 translation만큼 줄어들기 때문. 실제 D435i 스펙(뎁스-컬러 baseline ~15mm)과 부호가
    # 일치하는지도 함께 확인한다.
    depth = np.zeros((100, 100), dtype=np.float64)
    z = 0.5
    depth[50, 50] = z  # (cx=50, cy=50)가 depth_ppx/ppy와 일치 -> depth 좌표계에서 x=y=0
    aligned = align_depth_to_color(
        depth, depth_fx=600.0, depth_fy=600.0, depth_ppx=50.0, depth_ppy=50.0,
        color_fx=600.0, color_fy=600.0, color_ppx=50.0, color_ppy=50.0,
        rotation=np.eye(3), translation=np.array([0.015, 0.0, 0.0]),
        out_shape=(100, 100))
    expected_u = int(round(0.015 * 600.0 / z)) + 50  # x_color = 0 + 0.015 -> u = x*fx/z + ppx
    nz_v, nz_u = np.nonzero(aligned)
    assert len(nz_u) == 1
    assert nz_u[0] == expected_u
    assert nz_v[0] == 50
    assert aligned[50, expected_u] == pytest.approx(z)


def test_align_depth_keeps_closer_point_on_occlusion():
    # 시차 때문에 서로 다른 depth 픽셀 2개가 같은 컬러 픽셀에 겹쳐 투영되는 경우(occlusion),
    # 더 먼(배경) 점이 아니라 더 가까운(z가 작은) 점이 남아야 한다 - z-buffer 없이 나중에
    # 쓴 값이 그냥 덮어쓰면 순회 순서에 따라 배경이 전경을 뚫고 나오는 잘못된 결과가 나온다.
    depth = np.zeros((10, 10), dtype=np.float64)
    depth[5, 4] = 0.8  # 먼 점 (배경), z=0.8
    depth[5, 6] = 0.3  # 가까운 점 (전경), z=0.3 - translation 후 먼 점과 같은 컬러 픽셀로 겹친다
    # translation은 미터 단위라 픽셀 이동량은 z에 반비례(pixel_shift ~ shift/z)하므로,
    # 두 점의 z가 다르면 같은 shift라도 최종 컬러 픽셀이 다르게 밀린다. 두 점이 정확히
    # 같은 컬러 픽셀에 겹치는 shift를 x_far/0.8 == x_near/0.3 조건으로 풀면(x는 각각의
    # 재투영 후 컬러좌표계 x) shift = -0.0016 - 손으로 방정식을 풀어 구한 값.
    x_far, x_near = (4 - 5) * 0.8 / 600.0, (6 - 5) * 0.3 / 600.0
    shift = (0.3 * x_far - 0.8 * x_near) / 0.5
    aligned = align_depth_to_color(
        depth, depth_fx=600.0, depth_fy=600.0, depth_ppx=5.0, depth_ppy=5.0,
        color_fx=600.0, color_fy=600.0, color_ppx=5.0, color_ppy=5.0,
        rotation=np.eye(3), translation=np.array([shift, 0.0, 0.0]),
        out_shape=(10, 10))
    nz_v, nz_u = np.nonzero(aligned)
    assert len(nz_u) == 1  # 두 점이 같은 컬러 픽셀 하나로 겹쳤다
    assert aligned[nz_v[0], nz_u[0]] == pytest.approx(0.3)  # 더 가까운 점이 남음


def test_align_depth_no_valid_pixels_returns_zeros():
    depth = np.zeros((20, 20), dtype=np.float64)
    aligned = align_depth_to_color(
        depth, depth_fx=600.0, depth_fy=600.0, depth_ppx=10.0, depth_ppy=10.0,
        color_fx=600.0, color_fy=600.0, color_ppx=10.0, color_ppy=10.0,
        rotation=np.eye(3), translation=np.zeros(3), out_shape=(20, 20))
    assert np.all(aligned == 0.0)


def test_patch_median_ignores_holes():
    depth = np.full((100, 100), 0.5)
    depth[48:53, 48:53] = 0.0  # 중심에 뎁스 구멍
    z, ratio = patch_median_depth(depth, 50, 50, half=4)
    assert z == pytest.approx(0.5)  # 구멍 주변의 유효 픽셀로 복원
    assert 0.0 < ratio < 1.0


def test_patch_median_all_invalid():
    depth = np.zeros((100, 100))
    z, ratio = patch_median_depth(depth, 50, 50)
    assert z is None and ratio == 0.0


@pytest.mark.parametrize('angle', [0, 30, 45, 90, 120, 170])
def test_tool_axis_recovers_angle(angle):
    roi = _synthetic_roi(angle)
    axis_deg, rect, elongation = tool_axis_from_depth(roi, band_m=0.008, **FAKE_INTR)
    assert axis_deg is not None
    assert elongation > 1.3  # 140x30 막대는 장단축비가 뚜렷해야 함
    # 180도 주기 각도 차이
    diff = abs(axis_deg - angle) % 180
    assert min(diff, 180 - diff) < 3.0


def test_tool_axis_excludes_belt():
    """마스크가 벨트(1cm 아래)를 물지 않아야 장축이 막대를 따라간다 - band 8mm 검증."""
    roi = _synthetic_roi(45, belt_z=0.310, tool_z=0.300)
    axis_deg, _, _ = tool_axis_from_depth(roi, band_m=0.008, **FAKE_INTR)
    diff = abs(axis_deg - 45) % 180
    assert min(diff, 180 - diff) < 3.0


@pytest.mark.parametrize('angle', [0, 45, 90, 135])
def test_tool_axis_uses_depth_tilt_when_silhouette_ambiguous(angle):
    """XY 실루엣은 거의 정사각형(장단축비 낮음)이라도, 장축 방향 뎁스 기울기가 있으면
    3D PCA가 각도를 복원해야 한다 - 벨트 밖으로 걸쳐 기운 물체(사용자 실기 관찰) 대응.
    2D 이미지 모멘트만으로는 이 정보를 볼 수 없어 방향이 사실상 무작위로 튄다."""
    roi = _synthetic_tilted_roi(angle)
    axis_deg, _, elongation = tool_axis_from_depth(roi, band_m=0.008, **FAKE_INTR)
    assert axis_deg is not None
    assert elongation < 1.3  # 실루엣 자체는 정사각형에 가까워야(장단축비 낮음) 테스트 취지에 맞음
    diff = abs(axis_deg - angle) % 180
    assert min(diff, 180 - diff) < 10.0


def test_axis_smoother_wraparound():
    """0<->179도 경계에서 평균이 90도로 튀지 않고 경계 근처에 머물러야 한다."""
    s = AxisSmoother(alpha=0.5)
    s.update('k', 179.0)
    out = s.update('k', 1.0)
    assert out > 170 or out < 10


def test_axis_smoother_current_returns_state_without_update():
    """current()는 마지막 스무딩 각도를 상태 변경 없이 돌려준다 - 관측 불가 프레임에서
    직전 축을 유지(hold)하는 용도. 이력이 없으면 None."""
    s = AxisSmoother(alpha=0.5)
    assert s.current('k') is None
    s.update('k', 40.0)
    assert s.current('k') == pytest.approx(40.0)
    assert s.current('k') == pytest.approx(40.0)  # 반복 조회에도 상태 불변


def test_axis_smoother_reset():
    s = AxisSmoother(alpha=0.25)
    s.update('k', 10.0)
    s.reset('k')
    assert s.update('k', 90.0) == pytest.approx(90.0)  # 이력 없이 새 값 그대로


def test_axis_smoother_alpha_override_dampens_update():
    """alpha를 낮게 주면 관측치가 이전 값에서 덜 이동해야 한다(저신뢰 관측 억제)."""
    s = AxisSmoother(alpha=0.9)
    s.update('k', 0.0)
    damped = s.update('k', 90.0, alpha=0.05)
    assert damped < 10.0


def test_axis_smoother_reset_missing():
    s = AxisSmoother(alpha=0.5)
    s.update('a', 10.0)
    s.update('b', 20.0)
    s.reset_missing({'a'})  # 'b'는 이번 프레임에 안 보였다고 가정
    assert s.update('a', 15.0) != pytest.approx(15.0)  # 'a'는 이력 유지
    assert s.update('b', 15.0) == pytest.approx(15.0)  # 'b'는 이력 없이 새 값 그대로


def test_bbox_edge():
    assert is_bbox_at_edge(0, 100, 200, 200, 640, 480)
    assert is_bbox_at_edge(100, 100, 635, 200, 640, 480)
    assert not is_bbox_at_edge(100, 100, 200, 200, 640, 480)


def test_zyz_identity_and_flip():
    assert np.allclose(zyz_deg_to_rot(0, 0, 0), np.eye(3))
    # B=180: 그리퍼가 아래를 봄 - z축 반전
    assert np.allclose(zyz_deg_to_rot(0, 180, 0) @ [0, 0, 1], [0, 0, -1])


def test_posx_matrix_translation():
    T = posx_to_matrix([100.0, 200.0, 300.0, 0.0, 180.0, 0.0])
    assert np.allclose(T[:3, 3], [100, 200, 300])


def test_yaw_quaternion():
    qx, qy, qz, qw = yaw_deg_to_quaternion(90.0)
    assert (qx, qy) == (0.0, 0.0)
    assert qz == pytest.approx(np.sin(np.pi / 4))
    assert qw == pytest.approx(np.cos(np.pi / 4))
