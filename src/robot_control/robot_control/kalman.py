import numpy as np


class KalmanXYZV:
    """base_link 좌표 공구 위치용 등속 모델 칼만 필터. 상태=[x,y,z,vx,vy] (전체 계획.md 2.5절)."""

    def __init__(self, q_pos=1e-4, q_vel=1e-2, r_xy=1e-4, r_z=1e-4, p0_vel_reset=1.0):
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r_xy = r_xy
        self.r_z = r_z
        self.p0_vel_reset = p0_vel_reset
        self.x = np.zeros(5)
        self.P = np.eye(5)
        self._initialized = False

    def initialize(self, x, y, z):
        self.x = np.array([x, y, z, 0.0, 0.0])
        self.P = np.eye(5)
        self._initialized = True

    def _F(self, dt):
        F = np.eye(5)
        F[0, 3] = dt
        F[1, 4] = dt
        return F

    def _Q(self, dt):
        q = np.array([self.q_pos, self.q_pos, self.q_pos, self.q_vel, self.q_vel])
        return np.diag(q * max(dt, 1e-6))

    def predict(self, dt):
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def predict_position(self, lead_time):
        """상태를 바꾸지 않고 lead_time 이후 위치만 외삽해서 반환한다."""
        px, py, pz, vx, vy = self.x
        return np.array([px + vx * lead_time, py + vy * lead_time, pz])

    def update_xyz(self, meas_xyz):
        H = np.zeros((3, 5))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        R = np.diag([self.r_xy, self.r_xy, self.r_z])
        return self._update(H, R, np.asarray(meas_xyz, dtype=float))

    def update_xy_only(self, meas_xy):
        H = np.zeros((2, 5))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        R = np.diag([self.r_xy, self.r_xy])
        return self._update(H, R, np.asarray(meas_xy, dtype=float))

    def _update(self, H, R, z):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(5)
        self.P = (I - K @ H) @ self.P
        return float(np.linalg.norm(y[:2]))

    def reset_velocity_covariance(self):
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
        return float(self.P[3, 3] + self.P[4, 4])
