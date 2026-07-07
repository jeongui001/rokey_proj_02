# handover_approach의 RT/speedl_stream 요소 제거 (movel 구현은 범위 밖)

> 이 문서는 같은 이름으로 먼저 작성됐던 "movel 기반 단발성 이동 전체 구현"
> 설계를 대체한다. 사용자 확인: 실제 movel 로직(목표점 계산, 이동 실행)은
> 사용자가 직접 구현할 예정이라 이번 작업 범위가 아니다 - 이번엔 더 이상
> 안 쓰이게 될 RT/speedl_stream 관련 코드를 걷어내고, 그 자리에 TODO 스텁만
> 남긴다.

## 배경

`handover_approach`는 2026-07-07 RT 세션 제거 작업에서 `servo_pick`과 함께
`_run_rt_tracking`(비-RT `speedl_stream` 연속 속도 스트리밍 + `SpeedlWatchdog`)으로
마이그레이션됐다. 그런데 손 근처에서 멈추기만 하면 되고 정밀한 실시간 추적이
필요 없으므로, 이 방식 자체가 과하다 - 사용자가 나중에 `movel`(점대점 이동)로
직접 구현할 예정이다.

## 목적

- `handover_approach`가 `_run_rt_tracking`/`speedl_stream`/`SpeedlWatchdog`를
  전혀 쓰지 않게 한다.
- 액션 통신 구조(goal 수락/거부, safety_state·hardware_ready 게이트, Result 반환
  형식)는 그대로 유지해 나중에 movel 로직을 그 자리에 채워 넣기 쉽게 한다.
- 더 이상 쓰이지 않게 되는 `HandApproachServo`와 관련 코드·파라미터·테스트를
  완전히 제거한다.
- `robot_control_params.yaml`의 대응 오버라이드(`v_max`/`kp_xy`/`t_lost_s`)도
  정리해 죽은 설정이 남지 않게 한다.

## 범위 밖

- movel 목표점 계산, `move_line` 호출, dry-run 분기 등 실제 접근 로직 구현 -
  사용자가 직접 함.
- `servo_pick`은 전혀 건드리지 않는다 - `_run_rt_tracking`/`speedl_stream`/
  `SpeedlWatchdog`는 servo_pick에 그대로 남는다(단일 호출자가 되지만 함수
  이름·구조는 유지).
- `handover_hold`(compliance 기반) - 무관, 변경 없음.
- vision_node의 `_track_hand` - 이미 MediaPipe로 구현돼 있음을 확인했으나
  이 작업과 무관, 손대지 않는다.

## 사전 점검 결과 (오늘 변경분과의 모순 확인)

- `robot_control_params.yaml:77-87`의 `handover_approach:` 블록에
  `v_max: 0.15`, `kp_xy: 1.0`, `t_lost_s: 0.5`가 명시적으로 오버라이드돼 있음 -
  이 파라미터들의 `declare_parameter` 호출을 지우면 대응 파라미터가 없어져
  아무 효과 없는 죽은 설정이 됨. **이 작업에서 같이 제거한다.**
- `task_manager/task_flow.py`(`_handle_approach_hand_result`)와 관련 테스트는
  `handover_approach` 액션의 `result.success`/`result.message`만 사용하고
  feedback 문자열이나 다른 Result 필드는 보지 않음 - TODO 스텁이
  `success=False`를 반환해도 기존 `_enter_fault()` 경로가 정상 동작하므로
  호환됨. task_manager 쪽 변경 불필요.
- launch 파일에는 `handover_approach` 관련 파라미터 참조 없음 - 추가 점검 불필요.
- `task_executor.py`의 `from geometry_msgs.msg import PoseStamped` import는
  `_compute_hand_pose_tcp_offset`과 `_execute_handover_approach`의
  `_run_rt_tracking` 호출(`message_type=PoseStamped`)에서만 쓰였음 - 둘 다
  삭제되므로 이 import도 함께 제거해야 orphan import가 안 남는다.

## 변경 내용

### `robot_control/robot_control/servo_loop.py`
- `HandApproachServo`, `HandApproachState` 클래스 전체 삭제.
  (`ServoLoop`/`ServoState`/`ServoCommand`는 servo_pick이 쓰므로 그대로 유지)

### `robot_control/test/test_servo_loop.py`
- `HandApproachServo`/`HandApproachState` 관련 테스트 12개 전체 삭제:
  `test_hand_approach_initial_state_is_tracking`,
  `test_hand_approach_step_with_no_pose_yet_returns_zero_command`,
  `test_hand_approach_step_commands_velocity_toward_positive_error`,
  `test_hand_approach_step_commands_velocity_toward_negative_error`,
  `test_hand_approach_step_clips_velocity_to_v_max`,
  `test_hand_approach_should_stop_within_distance`,
  `test_hand_approach_should_not_stop_outside_distance`,
  `test_hand_approach_should_stop_false_before_any_pose`,
  `test_hand_approach_should_abort_timeout`,
  `test_hand_approach_should_abort_lost_when_no_update_within_t_lost`,
  `test_hand_approach_should_abort_diverged_when_error_grows`,
  `test_hand_approach_should_abort_none_when_converging`.
  Import 목록에서도 `HandApproachServo`/`HandApproachState` 제거.

### `robot_control/robot_control/task_executor.py`
- `from geometry_msgs.msg import PoseStamped` import 제거.
- `_on_hand_pose_during_approach`, `_handover_approach_tick`,
  `_validate_handover_approach_command`, `_compute_hand_pose_tcp_offset`
  메서드 전체 삭제.
- `_execute_handover_approach`를 다음과 같이 축소(통신 구조는 그대로,
  본문만 TODO 스텁):

```python
def _execute_handover_approach(self, goal_handle):
    if self.safety_state != SafetyState.NORMAL:
        return self._finish_tracking_result(
            goal_handle, 'ABORT',
            f'handover_approach rejected - safety_state={self.safety_state}')
    if (self.hardware_enabled
            and not self.get_parameter('handover_approach.hardware_ready').value):
        return self._finish_tracking_result(
            goal_handle, 'ABORT',
            'handover_approach rejected - handover_approach.hardware_ready=false')

    # TODO: movel 기반 단발성 접근 구현 예정 - RT/speedl_stream 미사용.
    # /vision/hand_pose를 한 번만 읽어 현재 TCP->손 방향으로
    # stop_distance_m만큼 못 미친 지점까지 movel로 이동한다(재계산 없음).
    return self._finish_tracking_result(
        goal_handle, 'ABORT', 'handover_approach not yet implemented')
```

### `robot_control/robot_control/robot_control_node.py`
- `self.hand_approach_servo = HandApproachServo(...)` 생성 코드 삭제.
- 다음 파라미터 선언 삭제: `handover_approach.v_max`, `handover_approach.kp_xy`,
  `handover_approach.t_lost_s`, `handover_approach.diverge_factor`,
  `handover_approach.diverge_window`, `handover_approach.control_period_s`,
  `handover_approach.speedl_acc_trans_mm_s2`,
  `handover_approach.speedl_acc_rot_deg_s2`, `handover_approach.watchdog_timeout_s`.
- 유지: `handover_approach.hardware_ready`, `handover_approach.stop_distance_m`,
  `handover_approach.timeout_s`, `handover_approach.hand_pose_frame_id`
  (나중에 movel 구현 시 재사용).

### `robot_control/config/robot_control_params.yaml`
- `handover_approach:` 블록에서 `v_max: 0.15`, `kp_xy: 1.0`, `t_lost_s: 0.5`
  세 줄 제거.
- 75-76행 주석("PBVS 서보잉" 전제)을 TODO 스텁 상태에 맞게 갱신.

### `robot_control/test/test_robot_control_node.py`
**삭제**(스트리밍/PBVS 전제 테스트) - 플랜 작성 중 2차 점검으로 다음 6개를
추가 발견함(최초 스펙에서 누락됨): `_hand_pose_msg` 헬퍼와
`test_compute_hand_pose_tcp_offset_computes_hand_minus_tcp`,
`test_compute_hand_pose_tcp_offset_rejects_wrong_frame_id`,
`test_compute_hand_pose_tcp_offset_rejects_nan_inf_position`,
`test_compute_hand_pose_tcp_offset_returns_none_when_tcp_lookup_fails`,
`test_on_hand_pose_during_approach_ignores_message_when_offset_computation_fails`,
`test_on_hand_pose_during_approach_forwards_computed_offset` (전부
`_compute_hand_pose_tcp_offset`/`_on_hand_pose_during_approach` 삭제와 함께
사라져야 하는 테스트들). 아래 목록에 이어서:
`test_validate_handover_approach_command_rejects_nan_inf`,
`test_validate_handover_approach_command_rejects_over_v_max`,
`test_validate_handover_approach_command_accepts_within_limits`,
`test_handover_approach_tick_continue`, `test_handover_approach_tick_stop`,
`test_handover_approach_tick_abort`,
`test_execute_handover_approach_success_stops_without_gripper_action`,
`test_execute_handover_approach_aborts_on_abort_reason`,
`test_execute_handover_approach_cancel_mid_loop_calls_canceled`,
`test_execute_handover_approach_sets_and_clears_tcp_tracking_active`,
`test_handover_approach_publishes_speedl_with_own_param_prefix`.

**유지**(게이트 로직 - TODO 스텁에서도 그대로 성립):
`test_goal_callback_rejects_handover_approach_when_hardware_ready_false`,
`test_goal_callback_accepts_handover_approach_in_dry_run`,
`test_execute_handover_approach_rejected_when_hardware_ready_false`,
`test_execute_handover_approach_rejected_when_safety_state_not_normal`.

**추가**(TODO 스텁 자체의 동작 - 게이트를 통과하면 항상 실패로 끝난다는 걸
명시적으로 검증):
```python
def test_execute_handover_approach_returns_not_implemented_when_gates_pass(node):
    gh = FakeGoalHandle(_goal('handover_approach'))

    result = node._execute_handover_approach(gh)

    assert gh.aborted is True
    assert result.success is False
    assert result.message == 'handover_approach not yet implemented'
```

## 자기 점검

- 삭제 대상 심볼(`HandApproachServo` 등)이 `task_manager`/`vision_node` 등
  다른 패키지에서 참조되지 않음을 grep으로 확인함 - 크로스 패키지 영향 없음.
- yaml 오버라이드 제거가 코드의 파라미터 삭제와 정확히 짝을 이룸 - 죽은
  설정이 남지 않음.
- `PoseStamped` import 제거가 orphan import를 만들지 않는지 확인함(다른
  용도로 안 쓰임).
- 유지되는 4개 파라미터(`hardware_ready`/`stop_distance_m`/`timeout_s`/
  `hand_pose_frame_id`)는 모두 TODO로 남긴 미래 movel 구현이 실제로 쓸 것들이라
  지금 지우지 않는 것이 맞음.
