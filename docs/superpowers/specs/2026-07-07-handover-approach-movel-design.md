# handover_approach를 movel 기반 단발성 이동으로 전환하는 설계

## 배경

`handover_approach`(handover_safe 도착 후 작업자 손에 접근)는 현재 `servo_pick`과
동일한 `_run_rt_tracking`(비-RT `speedl_stream` 연속 속도 스트리밍 + `SpeedlWatchdog`)을
공유해서 쓰고 있다(2026-07-07 RT 세션 제거 작업에서 그렇게 마이그레이션됨).

그런데 `handover_approach`는 손 근처에서 멈추기만 하면 되고 grasp처럼 정밀한
실시간 추적/제어가 필요 없다 - "그냥 movel로 처리"해도 충분하다. 이 설계는
`handover_approach`를 연속 속도 서보잉에서 **단발성 movel(점대점 Cartesian 이동)**
로 전환한다.

## 목적

- `handover_approach`가 더 이상 `_run_rt_tracking`/`speedl_stream`/`SpeedlWatchdog`를
  쓰지 않게 한다 - `move_named`와 같은 단발성 이동 패턴으로 전환.
- `/vision/hand_pose`를 **한 번만** 읽어 목표점을 계산하고, 그 이후 손 위치가
  바뀌어도 재계산하지 않는다(사용자 확정: "손 위치가 바뀌어도 첫번째 계산된
  손을 기준으로 이동").
- 목표점은 손 위치가 아니라 "현재 TCP→손 방향으로 `stop_distance_m`만큼 못
  미친 지점"이다(충돌 방지).
- 더 이상 쓰이지 않게 되는 `HandApproachServo`(PBVS 속도제어)와 관련 파라미터를
  완전히 제거한다.

## 범위 밖

- `servo_pick`은 그대로 `_run_rt_tracking`/`speedl_stream`/`SpeedlWatchdog`를 계속
  쓴다 - grasp에는 정밀한 연속 추적이 필요하므로 변경 없음.
- `handover_hold`(compliance 기반, 당김 감지) - 이 작업과 무관, 변경 없음.
- 이동 중 손 위치 재추적/재타겟팅 - 명시적으로 범위 밖(사용자 확정).
- vision_node의 `_track_hand`(현재 `NotImplementedError`) 구현 - 별개 작업.
  `handover_approach.hardware_ready`는 이 구현 이후에도 여전히 좌표계/오프셋
  검증이 필요하므로 계속 `False` 기본값을 유지한다.

## 아키텍처

`_execute_handover_approach`(task_executor.py)를 `_run_rt_tracking` 호출에서
`_execute_move_named`와 같은 형태의 단발성 흐름으로 재작성한다:

1. **손 위치 1회 대기**: `/vision/hand_pose`(`PoseStamped`)를 임시 구독해 첫
   메시지를 기다린다. 데드라인은 기존 `handover_approach.timeout_s`를 그대로
   재사용한다(손 대기 + 이동 전체를 아우르는 하나의 데드라인). 데드라인 초과,
   취소, 안전상태 이상 시 각각 ABORT/CANCELED로 종료하고 구독을 정리한다.
   수신된 메시지는 기존 `_compute_hand_pose_tcp_offset`이 하던 것과 동일하게
   `frame_id == handover_approach.hand_pose_frame_id`(base_link) 검증과
   NaN/Inf 검증을 거친다 - 실패하면 이번 시도 자체를 ABORT 처리한다(재시도
   없음, 손 위치를 한 번만 쓰는 설계이므로).
2. **목표점 계산**: 현재 TCP 위치(`_get_current_tcp_posx()`, mm)와 검증된 손
   위치(m, base_link)로 3D 유클리드 거리를 구한다.
   - `distance <= stop_distance_m`이면 movel을 아예 생략하고 바로 ARRIVED
     (이미 충분히 가까움).
   - 아니면 `target = tcp + unit(hand - tcp) * (distance - stop_distance_m)`
     (m 단위로 계산 후 mm로 변환). orientation(rx,ry,rz)은 현재 TCP 값을 그대로
     사용한다(손을 향해 회전하지 않음 - orientation 의미가 아직 정의 안
     됐다는 기존 제약과 동일).
3. **movel 실행**: `doosan_driver.move_line()`(이미 정의돼 있으나 미사용이던
   메서드, `MoveLine` 서비스 래퍼)을 한 번 호출한다. 취소/안전상태 처리는
   `_call_move_with_cancel`이 기존과 동일하게 담당한다(폴링 기반, 새 로직
   불필요).
4. 이동 완료 시 ARRIVED, 실패/취소/타임아웃 시 각각 대응하는 outcome을
   반환한다. 재계산·재시도 루프는 없다(1회 실패하면 그걸로 이 시도는 끝 -
   task_manager가 필요하면 액션을 재요청).

## 목표점 계산 상세

```
tcp = _get_current_tcp_posx()  # [x,y,z,rx,ry,rz], mm/deg
hand = 검증된 hand_pose.position  # m, base_link

tcp_m = tcp[:3] / 1000.0
error = hand - tcp_m  # m
distance = |error|

if distance <= stop_distance_m:
    ARRIVED (movel 생략)
else:
    target_m = tcp_m + (error / distance) * (distance - stop_distance_m)
    target_mm = target_m * 1000.0
    target_pos6 = [*target_mm, tcp[3], tcp[4], tcp[5]]  # orientation은 현재 TCP 유지
```

## 제거 대상

- `robot_control/robot_control/servo_loop.py`의 `HandApproachServo`,
  `HandApproachState` 클래스 전체 삭제 (`ServoLoop`/`ServoState`/`ServoCommand`는
  servo_pick이 계속 쓰므로 그대로 유지).
- `robot_control/test/test_servo_loop.py`의 `HandApproachServo` 관련 테스트
  전체 삭제(`test_hand_approach_*` 전부).
- `robot_control_node.py`의 파라미터: `handover_approach.v_max`,
  `handover_approach.kp_xy`, `handover_approach.t_lost_s`,
  `handover_approach.diverge_factor`, `handover_approach.diverge_window`,
  `handover_approach.control_period_s`, `handover_approach.speedl_acc_trans_mm_s2`,
  `handover_approach.speedl_acc_rot_deg_s2`, `handover_approach.watchdog_timeout_s`
  전부 삭제(전부 handover_approach에서만 쓰였음, servo_pick 쪽 동명 파라미터는
  별개라 영향 없음).
- `task_executor.py`의 `self.hand_approach_servo` 초기화, `_on_hand_pose_during_approach`,
  `_handover_approach_tick`, `_validate_handover_approach_command`,
  `_compute_hand_pose_tcp_offset` 삭제(전부 handover_approach 전용이었음).
- `_run_rt_tracking` 호출부에서 handover_approach 관련 호출 제거 - 이제
  `_run_rt_tracking`은 servo_pick만 호출한다. 함수 이름/구조는 그대로 유지한다
  (여전히 "물체 추적 실행 루프"로서 정확한 이름이고, 단일 호출자로 줄었다고
  구태여 리네이밍하지 않는다).
- `robot_control_node.py`의 `self.hand_approach_servo = HandApproachServo(...)`
  생성 코드 삭제.

## 추가 대상

`robot_control_node.py`에 새 파라미터:
- `handover_approach.move_vel_mm_s` = 150.0 (기존 `v_max`=0.15m/s와 동일한
  값 재사용 - 사람에게 접근하는 속도로 이미 협의된 값)
- `handover_approach.move_vel_deg_s` = 20.0 (orientation을 유지하므로 회전은
  거의 안 쓰이지만 movel 호출에 필요한 값 - 낮은 기본값)
- `handover_approach.move_acc_mm_s2` = 150.0
- `handover_approach.move_acc_deg_s2` = 60.0

유지되는 기존 파라미터: `handover_approach.hardware_ready`,
`handover_approach.stop_distance_m`, `handover_approach.timeout_s`,
`handover_approach.hand_pose_frame_id`.

## 새 `_execute_handover_approach` 흐름 (의사코드)

```python
def _execute_handover_approach(self, goal_handle):
    if self.safety_state != NORMAL: -> ABORT
    if hardware_enabled and not handover_approach.hardware_ready: -> ABORT

    hand_pose = 임시 구독으로 첫 PoseStamped 대기(timeout_s, 취소/안전상태 감시)
    if 타임아웃/취소/안전상태이상: 해당 outcome으로 종료 (구독 정리)
    if frame_id 불일치 또는 NaN/Inf: ABORT

    tcp = self._get_current_tcp_posx()
    if tcp is None: ABORT (TCP 조회 실패)

    distance, target_pos6 = 목표점 계산(위 로직)
    if distance <= stop_distance_m:
        return ARRIVED (movel 생략)

    success = self._doosan.move_line(
        goal_handle, target_pos6,
        vel2=[move_vel_mm_s, move_vel_deg_s],
        acc2=[move_acc_mm_s2, move_acc_deg_s2], ref=0)
    return ARRIVED or 실패/취소에 맞는 outcome
```

hardware_enabled=False(dry-run)일 때는 실제 서비스 호출 없이 상태 흐름만
검증하도록, `_move_joint`와 병렬 구조인 `_move_line` 헬퍼를 새로 추가한다:

```python
def _move_line(self, goal_handle, pos6, vel2, acc2, ref=0) -> bool:
    if not self.hardware_enabled:
        return self._dry_run_move(goal_handle)  # 기존 헬퍼 그대로 재사용
    if self._doosan is None:
        self.get_logger().error('DoosanDriver가 초기화되지 않았습니다 - move_line 실패')
        return False
    return self._doosan.move_line(
        goal_handle, pos6, vel2, acc2, ref=ref,
        radius_mm=self.get_parameter('move.blend_radius_mm').value,
        sync_type=self.get_parameter('move.sync_type').value,
        poll_interval_s=self.get_parameter('move.poll_interval_s').value,
        timeout_s=self.get_parameter('move.timeout_s').value)
```

`move.blend_radius_mm`/`move.sync_type`/`move.poll_interval_s`/`move.timeout_s`는
기존 `_move_joint`가 쓰는 것과 동일한 파라미터를 그대로 공유한다(movel 전용
새 파라미터를 추가하지 않는다 - 이 값들은 이동 방식과 무관한 통신/폴링
설정이므로).

## 테스트 마이그레이션

- `test_servo_loop.py`: `HandApproachServo`/`HandApproachState` 관련 테스트
  전체 삭제.
- `test_robot_control_node.py`: 기존 `test_execute_handover_approach_*` 테스트들
  (RT 스트리밍 기반 - tick/step 스텁, feedback 개수 검증 등)을 새 단발성 흐름에
  맞게 재작성. 목표점 계산 로직은 순수 함수로 뽑아 유닛 테스트 가능하게 한다
  (예: `_compute_handover_approach_target(tcp, hand_pos, stop_distance_m) ->
  (distance, target_pos6) | None`).

## 자기 점검

- 목표점 계산식의 방향/부호가 "TCP에서 손 쪽으로, stop_distance_m만큼 못
  미친 지점"이라는 요구사항과 일치하는지 재확인함 - `unit(hand-tcp) *
  (distance - stop_distance_m)`이 TCP로부터 손 방향으로 정확히 그만큼 이동한
  지점이 맞음.
- 새 파라미터 기본값(150.0, 20.0, 150.0, 60.0)은 모두 근거 있음(기존 v_max
  재사용 또는 기존 코드베이스의 유사 파라미터 관례 - speedl_acc_rot_deg_s2=60.0
  패턴).
- servo_pick 쪽 파라미터/코드는 이 문서 어디에도 손대지 않음 - 범위 경계 명확.
