class RG2Client:
    """OnRobot RG2 그리퍼를 Modbus TCP(Compute Box)로 제어하는 클라이언트."""

    def __init__(self, ip: str, port: int = 502):
        self.ip = ip
        self.port = port

    def open(self) -> None:
        """RG2를 완전 개방한다."""
        raise NotImplementedError('RG2Client.open 구현 필요 (Modbus TCP 레지스터 쓰기)')

    def close(self, width_mm: float, force_n: float) -> None:
        """지정한 폭(mm)·힘(N)으로 RG2를 폐합한다."""
        raise NotImplementedError('RG2Client.close 구현 필요 (Modbus TCP 레지스터 쓰기)')

    def get_state(self):
        """(width_mm: float, grip_detected: bool) 튜플을 반환한다."""
        raise NotImplementedError('RG2Client.get_state 구현 필요 (Modbus TCP 레지스터 읽기)')
