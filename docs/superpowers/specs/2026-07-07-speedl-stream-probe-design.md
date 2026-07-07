# speedl_stream 비-RT 서보잉 실현가능성 검증 스크립트

## 배경

`robot_control`의 servo_pick은 현재 두산 RT 제어(`ConnectRtControl`/`StartRtControl` +
`SpeedlRtStream`)로 시각 서보잉 속도 명령을 흘려보내도록 설계돼 있다. 그런데
`dsr_msgs2`에는 RT 세션 없이 쓸 수 있는 **비-RT 스트리밍 토픽**(`speedl_stream`,
`servol_stream`)이 별도로 존재하며, 드라이버(`dsr_controller2.cpp`)에서 이 토픽의
콜백이 곧바로 `Drfl->speedl()`/`Drfl->servol()`을 호출한다 — DRL이 "환경 적응형
동작"으로 분류하는, 외부에서 반복 호출해 연속적인 움직임을 만들도록 설계된 함수다.

사용자가 이전에 노트북에서 `servol`을 반복 호출했을 때 `movel`과 달리 부드럽게
움직이는 걸 이미 확인한 바 있다. RT 세션(1초 워치독, UDP 채널, Connect/Start/Stop/
Disconnect 생명주기)을 걷어내고 이 비-RT 스트리밍으로 교체할 수 있는지 실제
하드웨어에서 검증하기 위한 최소 진단 스크립트를 만든다.

## 목적

다음을 실측으로 확인한다:
1. `speedl_stream`을 고정 주기로 반복 발행했을 때 실제로 부드럽게 움직이는지
2. 방향이 바뀌는 상황(오시레이션)에서도 부드러운지
3. 발행이 일시 중단됐다가 재개되면 로봇이 어떻게 반응하는지(정지/감속/유지)
4. `vel` 필드의 단위(mm/s 추정, 미검증)가 실제로 맞는지 — 작은 값으로 시작해
   위험 없이 확인
5. 로봇이 움직이는 도중에 그리퍼(RG2, Modbus TCP - 완전히 별개 경로)가 문제없이
   열고 닫히는지

## 범위 밖

- RT 세션 제거/`doosan_driver.py`·`task_executor.py` 리팩터링 (이 검증 결과를 보고
  별도로 결정)
- 자동화된 성공/실패 판정 (사람이 로봇을 보고 판단하는 수동 진단 도구)
- 좌표 변환/TCP 오프셋 계산 (이 스크립트는 base_link 원점 기준 순수 속도 명령만
  다룬다 — servo_pick의 실제 좌표계 검증과는 별개)

## 안전 설계

- **기본값은 매우 보수적**: 선속도 20mm/s(2cm/s 상당, mm/s 가정), 단일 축(기본 x),
  구간별 지속시간 3~5초.
- 실행 전 필수 확인: "실제 로봇이 움직입니다. Enable 스위치/펜던트를 쥐고 있어야
  합니다" 경고 후 인터랙티브로 `yes` 입력받기 전에는 아무 것도 발행하지 않는다.
- SIGINT/SIGTERM 수신 시 즉시 `vel=[0]*6`을 여러 번 발행한 뒤 종료 — 중단 시점에
  마지막 속도가 계속 유지된 채 스크립트만 죽는 상황을 막는다.
- `dsr_msgs2` import는 `doosan_driver.py`와 동일하게 함수 내부 지연 import로 처리.
- 각 단계 시작/종료를 콘솔에 배너로 출력해 로그와 실제 로봇 동작 타이밍을 맞춰볼
  수 있게 한다.
- 단위(mm/s vs m/s) 가정이 틀렸을 경우를 대비해 1단계 시작 직후 몇 초는 특히
  주의 깊게 관찰하도록 안내 문구를 출력한다.

## 단계 구성 (순서대로 실행)

1. **일정속도 지속 발행**: 지정 축으로 `vel=20mm/s` 고정값을 `period=0.02s`
   (50Hz)로 3초간 `speedl_stream`에 발행 → 끊김 여부 관찰
2. **방향 전환(오시레이션)**: 같은 축에서 부호를 1초 주기로 반전하며 4초간 발행
3. **명령 중단/재개**: 짧게 발행 → **의도적으로 발행 중단**(0.5s → 1s → 2s 순서로
   3회, 이 구간 동안은 절대 vel=0도 보내지 않고 "그냥 멈춤") → 재개. 각 구간
   전후 배너로 "지금부터 N초간 발행을 멈춥니다 - 로봇 반응을 관찰하세요" 안내
4. **명시적 정지**: 위 결과와 무관하게 `vel=0`을 여러 번 발행해 확실히 정지시킨
   뒤 그리퍼 단계로 진입
5. **이동 중 그리퍼 동작 확인**: 메인 스레드는 계속 일정 속도(1단계와 동일한
   작은 값)로 `speedl_stream`을 발행하고, **별도 스레드**에서 기존
   `RG2Client.close(grasp_width_mm, grasp_force_n)` → 1초 대기 →
   `RG2Client.open()`을 호출한다(둘 다 블로킹 폴링이라 메인 발행 루프와 분리
   필요). 그리퍼 스레드가 끝나면 메인 루프도 정지.
6. **최종 정지**: `vel=0` 여러 번 발행 후 노드 종료.

## 컴포넌트

- `robot_control/robot_control/tools/probe_speedl_stream.py` (신규, 단일 파일)
  - `build_phase_plan(axis, vel_mm_s, ...) -> list`: 1~3단계의 (label, vx,vy,...,duration)
    스케줄을 만드는 순수 함수 — 하드웨어/rclpy 의존 없음, 단위 테스트 대상
  - `main()`: argparse → 확인 프롬프트 → rclpy 노드 생성 → `speedl_stream` publisher
    생성 → 단계별 실행 → RG2Client 이동중 그리퍼 테스트 → 종료 시그널 핸들러
- `setup.py`의 `entry_points.console_scripts`에
  `probe_speedl_stream = robot_control.tools.probe_speedl_stream:main` 추가
- `robot_control/test/test_probe_speedl_stream.py` (신규): `build_phase_plan`만
  검증(오시레이션 부호 반전, pause 목록 순서·지속시간)

## CLI 인자 (기본값)

- `--robot-id` (dsr01), `--axis` (x), `--vel-mm-s` (20.0)
- `--acc-trans-mm-s2` (100.0), `--acc-rot-deg-s2` (100.0) — `SpeedlStream.acc`(2요소:
  병진/회전)에 그대로 대응. 목표속도(20mm/s)가 워낙 작아 100mm/s²면 5단계 이내에
  램프업되므로 "가속 급함" 위험은 낮다고 보고 잡은 시작값 — 실측 후 조정 가능.
- `--phase-duration-s` (3.0), `--osc-period-s` (1.0), `--osc-duration-s` (4.0)
- `--pause-durations-s` (0.5,1.0,2.0), `--period-s` (0.02)
- `--rg2-ip` (192.168.1.1), `--rg2-port` (502), `--rg2-gripper` (rg2)
- `--grasp-width-mm`, `--grasp-force-n` (사용자가 실측 대상에 맞게 지정, 기본값 없이
  필수 인자로 요구 — 임의로 추측하지 않는다)

## 테스트 계획

- `build_phase_plan`의 순수 로직만 pytest로 검증 (오시레이션 부호가 매 주기
  반전되는지, pause 단계가 요청한 순서/길이로 나오는지)
- 스크립트 전체는 실제 하드웨어 관찰이 성공 기준이므로 자동 테스트 대상 아님 —
  이 사실을 스크립트 상단 주석에 명시
