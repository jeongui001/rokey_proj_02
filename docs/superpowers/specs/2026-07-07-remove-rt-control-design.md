# RT 세션 제거 및 비-RT speedl_stream 전환 설계

## 배경

`probe_speedl_stream.py`(2026-07-07)로 실제 M0609에서 검증한 결과([[speedl-stream-watchdog-finding]] 메모리 참고):
1. 비-RT `speedl_stream`(드라이버가 곧바로 `Drfl->speedl()` 호출, RT 세션 불필요)도 RT 스트리밍과 동일하게 부드러운 연속 서보잉이 가능하다.
2. 비-RT `speedl`에는 RT의 1초 워치독 같은 자동 정지가 없다 — 발행이 끊겨도 로봇이 마지막 속도로 계속 움직인다.
3. `vel=0` 명령을 **단 한 번만** 발행해도 로봇이 멈춘 채 유지된다(실측 확인).

이를 근거로, `robot_control`의 `servo_pick`/`handover_approach`가 쓰는 두산 RT 제어(`ConnectRtControl`/`StartRtControl`/`StopRtControl`/`DisconnectRtControl` + `SpeedlRtStream`)를 완전히 제거하고, 비-RT `speedl_stream`으로 전환한다. RT가 제공하던 "명령 끊기면 자동 정지"는 애플리케이션 레벨의 데드맨스위치 워치독으로 대체한다.

## 목적

- RT 세션 생명주기 코드(연결/시작/종료/해제) 전부 제거 — 코드 단순화.
- `servo_pick`, `handover_approach` 둘 다 전환 (둘 다 `_run_rt_tracking`을 공유하므로 함께 처리).
- 별도 스레드로 도는 데드맨스위치 워치독 추가: 메인 서보 루프가 `timeout_s`(0.2초) 이내에 "살아있다" 신호(`pet()`)를 못 보내면, 워치독 스레드가 독립적으로 `vel=0`을 발행한다.
- 워치독은 같은 프로세스 내 스레드 기반이며, **프로세스 자체가 죽는 경우(kill -9, segfault 등)는 보호 범위 밖**이다(사용자 확인됨 — 외부 프로세스 감시자는 이번 범위 밖).

## 범위 밖

- 프로세스 레벨 외부 감시자(systemd watchdog 등) — 별도 논의 필요 시 차후 진행.
- `handover_hold`(compliance 기반, RT 스트리밍과 무관) — 변경 없음.
- 좌표계/TCP 오프셋 검증(`servo_pick.hardware_ready`/`handover_approach.hardware_ready`가 여전히 false인 이유) — 이번 작업과 무관한 별개 이슈이므로 게이트 자체는 유지.
- `probe_speedl_stream.py` 자체 — 이미 완료된 별도 산출물, 이번 작업에서 손대지 않음(다만 이 스크립트의 실측 결과가 이번 설계의 근거).

## 안전 설계 — `SpeedlWatchdog`

새 파일 `robot_control/robot_control/speedl_watchdog.py`:

```python
class SpeedlWatchdog:
    """메인 서보 루프와 별개 스레드에서 도는 데드맨스위치.

    루프가 pet()을 timeout_s 이내에 호출하지 않으면(예외로 루프가 죽거나,
    행(hang)이 걸린 경우) 워치독 스레드가 독립적으로 on_timeout()을 호출한다.
    rclpy에 의존하지 않는 순수 파이썬 클래스 - 하드웨어 없이 유닛 테스트 가능.
    """

    def __init__(self, timeout_s, on_timeout, poll_interval_s=0.05):
        ...

    def start(self) -> None:
        """워치독 스레드를 시작한다. last_pet을 현재 시각으로 초기화."""

    def pet(self) -> None:
        """메인 루프가 매 틱 호출 - '나 살아있다' 신호."""

    def stop(self) -> None:
        """워치독 스레드를 정지하고 join한다."""
```

- `on_timeout`은 `_run_rt_tracking`에서 `lambda: self._doosan.publish_speedl(ServoCommand())`(vel=0, 즉 정지)로 넘겨준다.
- 한 번 timeout이 발동하면 스레드는 종료한다(재시작하려면 `start()`를 다시 호출) — 반복적으로 vel=0을 계속 쏟아내지 않는다(실측상 1회로 충분하므로).
- `poll_interval_s`(기본 0.05초)로 나눠서 확인하므로, timeout 발동까지의 실제 지연은 `timeout_s + poll_interval_s` 이내.

## `_run_rt_tracking` 통합 (task_executor.py)

- 루프 시작 시(기존에 RT 세션을 열던 자리): `watchdog = SpeedlWatchdog(timeout_s=..., on_timeout=...)`, `watchdog.start()`.
- 루프 매 틱, `self._doosan.publish_speedl(command)` 발행 성공 직후: `watchdog.pet()`.
- `finally` 블록(기존에 RT 세션을 닫던 자리): `watchdog.stop()`.
- 기존 `_cleanup_stop_motion()`(MoveStop 서비스)은 그대로 유지 — 이건 RT/비-RT와 무관한 별개의 명시적 정지 경로(취소/안전상태 이상 시 사용).

## `doosan_driver.py` 변경

**제거**:
- `_ConnectRtControl`/`_StartRtControl`/`_StopRtControl`/`_DisconnectRtControl` import 및 관련 서비스 클라이언트(`_cli_connect_rt`/`_cli_start_rt`/`_cli_stop_rt`/`_cli_disconnect_rt`)
- `open_rt_session()`/`close_rt_session()` 메서드
- `_SpeedlRtStream` import, `_pub_speedl_rt` 퍼블리셔, `publish_speedl_rt()` 메서드

**추가**:
- `_SpeedlStream`(`dsr_msgs2.msg.SpeedlStream`) import
- `_pub_speedl` 퍼블리셔: `create_publisher(SpeedlStream, f'{prefix}/speedl_stream', 10)`
- `publish_speedl(command)`: `SpeedlStream(vel=[vx,vy,vz,0,0,yaw_rate], acc=[acc_trans, acc_rot], time=control_period_s)`를 발행. `probe_speedl_stream.py`의 `_run_publish_segment`와 동일한 메시지 구성 패턴을 따른다.

## 파라미터 변경 (`robot_control_node.py` + `robot_control_params.yaml`)

- **제거**: `servo_pick.rt_ip`, `servo_pick.rt_port` (UDP RT 세션 엔드포인트 전용, 더 이상 불필요)
- **이름 변경**: `servo_pick.rt_control_period_s` → `servo_pick.control_period_s`, `handover_approach.rt_control_period_s` → `handover_approach.control_period_s` (RT 아니므로 이름에서 "rt" 제거)
- **분리**: `servo_pick.speedl_acc`(6원소 리스트) → `servo_pick.speedl_acc_trans_mm_s2`(스칼라, 기본 200.0) + `servo_pick.speedl_acc_rot_deg_s2`(스칼라, 기본 60.0) — `SpeedlStream.acc`가 2원소(병진/회전)이기 때문
- **추가**: `servo_pick.watchdog_timeout_s`(기본 0.2), `handover_approach.watchdog_timeout_s`(기본 0.2)
- **유지**: `servo_pick.hardware_ready`/`handover_approach.hardware_ready` — 관련 주석에서 "RT 세션 시작 실패" 언급만 제거/갱신하고, 좌표계 미검증이라는 진짜 이유는 그대로 남긴다.

## 테스트 마이그레이션

- 새 `test/test_speedl_watchdog.py`: `SpeedlWatchdog`을 rclpy 없이 순수 로직으로 검증 — 짧은 `timeout_s`(예: 0.05초)와 가짜 `on_timeout` 콜백으로 (a) `pet()`이 계속 오면 콜백이 안 불림, (b) `pet()`이 멈추면 `timeout_s` 근처에서 콜백이 정확히 한 번 불림, (c) `stop()` 후에는 더 이상 콜백이 안 불림을 확인.
- `test_robot_control_node.py`(2550줄) 수정:
  - `_FakeDoosanDriver`에서 `open_rt_session`/`close_rt_session`/`publish_speedl_rt` 제거, `publish_speedl` 추가(같은 `publish_calls` 리스트에 기록).
  - 테스트 전반의 `node._open_rt_session = ...`/`node._close_rt_session = ...` 스텁 라인(약 20곳) 제거.
  - RT 세션 실패를 다루던 테스트(예: `test_servo_pick_aborts_when_start_rt_control_fails`, `close_rt_session_should_raise`/`close_rt_session_should_fail` 관련 테스트)는 삭제한다 — 더 이상 해당 실패 모드 자체가 존재하지 않음.
  - 워치독 관련 새 테스트 추가: `_run_rt_tracking` 실행 중 루프가 예외로 멈췄을 때(또는 pet이 중단됐을 때) 워치독이 `publish_speedl(vel=0)`을 호출하는지 — `RobotControlNode` 통합 레벨에서 `SpeedlWatchdog`의 `timeout_s`를 아주 짧게(예: 0.05초) 오버라이드해서 검증.

## 자기 점검(placeholder/모순/모호성)

- 모든 파라미터 기본값이 명시적 숫자로 정해져 있음(TBD 없음).
- `speedl_acc_trans_mm_s2`/`speedl_acc_rot_deg_s2` 기본값(200.0/60.0)은 기존 6원소 `speedl_acc` 기본값 `[200,200,200,60,60,60]`에서 병진/회전 각각의 대표값을 그대로 가져온 것 — 근거 있는 값(추측 아님).
- 범위 밖 항목(외부 프로세스 감시자, handover_hold, 좌표계 검증)을 명시해 스코프 경계가 분명함.
