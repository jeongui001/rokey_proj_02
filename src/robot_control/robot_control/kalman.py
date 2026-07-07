import numpy as np


class KalmanXYZV:
    """base_link 좌표 공구 위치용 등속 모델 칼만 필터. 상태=[x,y,z,vx,vy] (전체 계획.md 2.5절).

    z에는 속도 상태(vz)가 없다 - 컨베이어가 평면 위를 움직인다고 가정해서 x,y만
    등속 모델로 추적하고, z는 predict 단계에서 그대로 유지된다. 이 덕분에
    depth_valid=False 구간(update_xy_only만 호출)에서도 z가 "마지막 유효값 고정"
    동작을 자연스럽게 하게 된다 - 별도 분기 없이 상태 설계만으로 해결됨.
    """

    def __init__(self, q_pos=1e-4, q_vel=1e-2, r_xy=1e-4, r_z=1e-4, p0_vel_reset=1.0):
        self.q_pos = q_pos              # 프로세스 노이즈(위치) - 클수록 예측을 덜 믿음
        self.q_vel = q_vel               # 프로세스 노이즈(속도) - 등속 가정이 흔들릴 걸 감안해 위치보다 크게
        self.r_xy = r_xy                 # 측정 노이즈(x,y) - 클수록 새 관측을 덜 믿음
        self.r_z = r_z                   # 측정 노이즈(z)
        self.p0_vel_reset = p0_vel_reset  # 방향전환 감지 시 속도 공분산을 리셋할 값 (큰 값 = "속도 모른다"로 되돌림)
        self.x = np.zeros(5)            # 상태 [x, y, z, vx, vy]
        self.P = np.eye(5)              # 상태 공분산(추정 불확실성)
        self._initialized = False

    def initialize(self, x, y, z):
        """첫 ToolTrack을 받았을 때 한 번 호출 - 위치로 상태를 세팅하고 속도는 0(모름)으로 시작."""
        self.x = np.array([x, y, z, 0.0, 0.0])
        self.P = np.eye(5)
        self._initialized = True

    def _F(self, dt):
        """등속 모델의 상태전이행렬: x(t+dt) = x(t) + vx*dt (y도 동일), z는 그대로."""
        F = np.eye(5)
        F[0, 3] = dt
        F[1, 4] = dt
        return F

    def _Q(self, dt):
        """프로세스 노이즈 공분산 - dt에 비례해서 커진다(시간이 지날수록 예측 불확실성 누적)."""
        q = np.array([self.q_pos, self.q_pos, self.q_pos, self.q_vel, self.q_vel])
        return np.diag(q * max(dt, 1e-6))

    def predict(self, dt):
        """칼만 필터 예측 단계: x = F@x, P = F@P@F.T + Q. ToolTrack이 들어올 때마다
        update 직전에 호출한다(경과시간 dt만큼 앞으로 밀어서 현재 관측과 비교하기 위함)."""
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def predict_position(self, lead_time):
        """predict()와 달리 상태(self.x, self.P)를 바꾸지 않고, lead_time 이후 위치만
        읽기 전용으로 외삽해서 반환한다. servo_loop.step()이 이걸로 지연 보상
        (Δt_lat)된 목표점 p_ref를 계산한다(2.3절)."""
        px, py, pz, vx, vy = self.x
        return np.array([px + vx * lead_time, py + vy * lead_time, pz])

    def update_xyz(self, meas_xyz):
        """depth_valid=True일 때 - x,y,z 3개 다 관측했다고 보고 갱신. H는 상태에서
        위치 3개만 뽑아내는 관측행렬."""
        H = np.zeros((3, 5))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        R = np.diag([self.r_xy, self.r_xy, self.r_z])
        return self._update(H, R, np.asarray(meas_xyz, dtype=float))

    def update_xy_only(self, meas_xy):
        """depth_valid=False일 때 - z 행을 아예 뺀 H로 갱신하므로 z는 predict의 값(직전
        유효값) 그대로 남는다. x,y는 RGB 추적값으로 계속 갱신됨(2.7절 블라인드 구간 처리)."""
        H = np.zeros((2, 5))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        R = np.diag([self.r_xy, self.r_xy])
        return self._update(H, R, np.asarray(meas_xy, dtype=float))

    def _update(self, H, R, z):
        """칼만 필터 갱신 단계(표준식): innovation y = z - H@x, 칼만이득 K,
        상태/공분산 보정. 반환값(innovation의 xy 노름)은 서보 루프의 w 계산과
        공분산 리셋 판정에 재사용된다(2.5절 - "같은 innovation을 두 곳에 씀")."""
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(5)
        self.P = (I - K @ H) @ self.P
        return float(np.linalg.norm(y[:2]))

    def reset_velocity_covariance(self):
        """innovation이 innov_high를 넘었을 때(벨트 방향전환 등 등속 가정 위반) 호출.
        속도(vx,vy)에 대한 공분산만 크게 키워서 "속도는 다시 모른다"고 선언 -
        다음 관측들로 빠르게 재수렴하게 만드는 트릭."""
        for i in (3, 4):
            self.P[i, :] = 0.0
            self.P[:, i] = 0.0
            self.P[i, i] = self.p0_vel_reset

    @property
    def position(self):
        return self.x[:3].copy()

    @property
    def velocity(self):
        return self.x[3:5].copy()

    @property
    def velocity_covariance_trace(self):
        """속도 추정에 대한 전체 불확실성 크기(대각합). should_close()가 "필터가 충분히
        수렴했는지"를 판정하는 데 쓴다(2.6절 파지판정의 "필터 공분산 < 임계" 조건)."""
        return float(self.P[3, 3] + self.P[4, 4])
