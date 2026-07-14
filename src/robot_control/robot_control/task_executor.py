import math
import threading
import time

import rclpy

from handover_interfaces.action import RobotTask
from handover_interfaces.msg import HandTrack, ToolTrack

from robot_control.rg2_client import RG2Status
from robot_control.safety_monitor import FaultPrefix, SafetyState
from robot_control.servo_loop import ServoCommand, ServoState
from robot_control.speedl_watchdog import SpeedlWatchdog


class TaskExecutor:
    """RobotTask 5종을 실행하는 RobotControlNode용 mixin."""

    def _cleanup_stop_motion(self, stop_mode=None) -> bool:
        """M0609 MoveStop 정지를 시도한다. 실패/예외 시에도 절대 상위로 전파하지 않는다.
        stop_mode 미지정 시 safety.fault_stop_mode(빠른 정지)를 쓴다 - 원인이 불명확한
        예외/종료 경로의 기본값이다. 취소/일반 abort처럼 위험 상황이 아닌 정지는
        호출측이 safety.recoverable_stop_mode(부드러운 정지)를 명시적으로 넘긴다."""
        if not self.hardware_enabled or self._doosan is None:
            return True
        if stop_mode is None:
            stop_mode = self.get_parameter('safety.fault_stop_mode').value
        try:
            return bool(self._doosan.stop(stop_mode))
        except Exception as exc:
            self.get_logger().error(f'MoveStop cleanup 중 예외: {exc}')
            return False

    def _cleanup_disable_compliance(self) -> bool:
        """compliance 해제 (_disable_compliance 경계). 실패/예외 시에도 절대 상위로
        전파하지 않는다."""
        try:
            self._disable_compliance()
            return True
        except Exception as exc:
            self.get_logger().error(f'compliance 해제 cleanup 중 예외: {exc}')
            return False

    def _cleanup_destroy_subscription(self, sub) -> bool:
        """servo_pick 중 임시로 만든 ToolTrack subscription을 제거한다."""
        if sub is None:
            return True
        try:
            self.destroy_subscription(sub)
            return True
        except Exception as exc:
            self.get_logger().error(f'subscription 제거 cleanup 중 예외: {exc}')
            return False

    def _dry_run_move(self, goal_handle) -> bool:
        duration_s = self.get_parameter('move.dry_run_duration_s').value
        poll_interval_s = max(self.get_parameter('move.poll_interval_s').value, 0.001)
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            if goal_handle is not None and goal_handle.is_cancel_requested:
                return False
            if self.safety_state != SafetyState.NORMAL:
                return False
            time.sleep(min(poll_interval_s, max(deadline - time.monotonic(), 0.0)))
        return True

    def _move_joint(self, goal_handle, pos_deg6, vel, acc) -> bool:
        if not self.hardware_enabled:
            return self._dry_run_move(goal_handle)
        if self._doosan is None:
            self.get_logger().error('DoosanDriver가 초기화되지 않았습니다 - move_joint 실패')
            return False
        return self._doosan.move_joint(
            goal_handle, pos_deg6, vel, acc,
            radius_mm=self.get_parameter('move.blend_radius_mm').value,
            sync_type=self.get_parameter('move.sync_type').value,
            poll_interval_s=self.get_parameter('move.poll_interval_s').value,
            timeout_s=self.get_parameter('move.timeout_s').value)

    def _call_move_service(self, goal_handle=None, named_target='') -> bool:
        """고정 자세(movej) 이동을 수행한다."""
        if named_target:
            if named_target not in self._named_poses:
                self.get_logger().error(
                    f"알 수 없는 named pose '{named_target}' - 이동을 거부합니다 "
                    '(hardware_enabled 여부와 무관하게 dry-run에서도 거부됨).')
                return False
            pos = self._named_poses.get(named_target)
            if not pos:
                allow_dry_run = bool(
                    self.get_parameter('dry_run.allow_unconfigured_named_poses').value)
                if self.hardware_enabled or not allow_dry_run:
                    # hardware_enabled=true에서는 dry_run.allow_unconfigured_named_poses와
                    # 무관하게 빈 pose를 절대 허용하지 않는다.
                    self.get_logger().error(
                        f"named pose '{named_target}'의 관절값이 설정되지 않았습니다 "
                        f"(파라미터 named_poses.{named_target}). 이동을 수행하지 않습니다.")
                    return False
                self.get_logger().info(
                    f"[dry_run] named pose '{named_target}' 관절값 미설정 - "
                    'dry_run.allow_unconfigured_named_poses=true라서 dry-run 이동을 진행합니다.')
            else:
                # 값이 채워져 있다면(비어있지 않다면) hardware_enabled/dry-run 여부와
                # 무관하게 6개의 finite 숫자인지 검사한다 - 실제 관절 제한값은 근거가
                # 없어 추측하지 않지만, 개수/NaN·Inf는 여기서 확실히 걸러낸다.
                if len(pos) != 6 or not all(
                        isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(v) for v in pos):
                    self.get_logger().error(
                        f"named pose '{named_target}'의 관절값이 유효하지 않습니다 "
                        f"(정확히 6개의 finite 숫자여야 함, 현재 값={pos}) - 이동을 거부합니다.")
                    return False
            vel = self.get_parameter('move.vel_deg_s').value
            acc = self.get_parameter('move.acc_deg_s2').value
            success = self._move_joint(goal_handle, pos, vel, acc)
        else:
            self.get_logger().error('_call_move_service: named_target이 필요합니다.')
            return False
        # 이동 함수가 반환된 직후에도 안전상태를 다시 확인한다 - 이동 서비스가 성공
        # 응답을 반환하는 순간과 거의 동시에 Fault가 발생하는 경합으로 인해 Action이
        # 성공 처리되지 않도록 막는다.
        if success and self.safety_state != SafetyState.NORMAL:
            self.get_logger().warn(
                f'이동 완료 직후 안전상태 비정상({self.safety_state}) 감지 - 성공으로 처리하지 않습니다.')
            return False
        return success

    def _is_gripper_already_open(self) -> bool:
        """그리퍼가 이미 완전히 열린 폭이면 True. get_state() 통신 자체가 실패해
        0.0을 반환하는 경우까지 "이미 열림"으로 착각하면 안 되므로, 상태 조회가
        불확실할 때는 False를 반환해 open()을 그대로 시도하게 한다."""
        if not self.hardware_enabled:
            return False  # dry-run에서는 매번 그대로 시도해도 비용이 없다
        gripper = self.rg2_client.gripper
        if gripper not in self.rg2_client.MAX_WIDTH_MM:
            return False
        width_mm, _ = self.rg2_client.get_state()
        max_width = self.rg2_client.MAX_WIDTH_MM[gripper]
        tolerance = self.get_parameter('rg2.open_width_tolerance_mm').value
        return width_mm >= max_width - tolerance

    def _execute_move_named(self, goal_handle):
        result = RobotTask.Result()
        if self.safety_state != SafetyState.NORMAL:
            goal_handle.abort()
            result.success = False
            result.message = f'move_named rejected - safety_state={self.safety_state}'
            return result
        named_target = goal_handle.request.named_target
        try:
            if named_target in ('home', 'watch'):
                # home으로 이동하기 전, 그리고 watch(물건을 가지러 가기 시작점)로
                # 이동하기 전에는 그리퍼가 열린 상태임을 보장한다 - 이전에 어떤
                # 경로로 왔든(수동 이동, pick 중단, 검증 실패 등) 그리퍼가 물체를
                # 쥔 채로 다음 작업을 시작하지 않도록 하는 안전/리셋 동작이다
                # (_execute_release_and_retry의 RG2 open 처리와 동일한 패턴).
                # 이미 열려있으면 불필요한 재통신(및 그로 인한 오탐 FAULT 위험)을
                # 피해 생략한다 - _is_gripper_already_open() 참고.
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = f'move_named({named_target}) canceled before opening gripper'
                    return result
                if self.safety_state != SafetyState.NORMAL:
                    goal_handle.abort()
                    result.success = False
                    result.message = (
                        f'move_named({named_target}) aborted before opening gripper - '
                        f'safety_state={self.safety_state}')
                    return result
                if not self._is_gripper_already_open():
                    if not self.rg2_client.open(goal_handle=goal_handle):
                        if self.rg2_client.last_status == RG2Status.CANCELED:
                            goal_handle.canceled()
                            result.success = False
                            result.message = f'move_named({named_target}) canceled during RG2 open'
                            return result
                        detail = (
                            f'move_named({named_target}) RG2 open 실패'
                            f'(status={self.rg2_client.last_status}) - 안전을 위해 FAULT 처리')
                        self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                        goal_handle.abort()
                        result.success = False
                        result.message = detail
                        return result
            success = self._call_move_service(
                goal_handle=goal_handle, named_target=named_target)
        except Exception as exc:  # 통신 오류 등 예외 발생 시 성공을 반환하지 않는다
            self.get_logger().error(f'move_named 실행 중 예외: {exc}')
            goal_handle.abort()
            result.success = False
            result.message = f'move_named exception: {exc}'
            return result
        if not success and goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success = False
            result.message = f'move_named({goal_handle.request.named_target}) canceled'
            return result
        if success:
            goal_handle.succeed()
            result.success = True
        else:
            goal_handle.abort()
            result.success = False
            result.message = f'move_named({goal_handle.request.named_target}) failed'
        return result

    def _execute_release_and_retry(self, goal_handle):
        result = RobotTask.Result()
        if self.safety_state != SafetyState.NORMAL:
            goal_handle.abort()
            result.success = False
            result.message = f'release_and_retry rejected - safety_state={self.safety_state}'
            return result
        try:
            # RG2를 실제로 열기 직전에 취소/안전 상태를 다시 확인한다 - 그 사이 취소나
            # Fault가 발생했다면 그리퍼를 열지 않는다.
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = 'release_and_retry canceled before opening gripper'
                return result
            if self.safety_state != SafetyState.NORMAL:
                goal_handle.abort()
                result.success = False
                result.message = (
                    f'release_and_retry aborted before opening gripper - '
                    f'safety_state={self.safety_state}')
                return result
            open_ok = self.rg2_client.open(goal_handle=goal_handle)
            if not open_ok:
                if self.rg2_client.last_status == RG2Status.CANCELED:
                    # 실행 중 취소가 정상적으로 확인되어 command=8(stop)으로 멈췄다 -
                    # FAULT가 아니라 canceled()로 마무리한다.
                    goal_handle.canceled()
                    result.success = False
                    result.message = 'release_and_retry canceled during RG2 open'
                    return result
                # RG2 open 실패(잘못된 설정/통신 오류/busy timeout) - 물체를 놓쳤는지
                # 불확실한 채로 이동을 계속하지 않는다.
                detail = (
                    f'release_and_retry RG2 open 실패'
                    f'(status={self.rg2_client.last_status}) - 안전을 위해 FAULT 처리')
                self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                goal_handle.abort()
                result.success = False
                result.message = detail
                return result
            success = self._call_move_service(
                goal_handle=goal_handle, named_target='watch')
        except Exception as exc:
            self.get_logger().error(f'release_and_retry 실행 중 예외: {exc}')
            goal_handle.abort()
            result.success = False
            result.message = f'release_and_retry exception: {exc}'
            return result
        if not success and goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success = False
            result.message = 'release_and_retry canceled'
            return result
        if success:
            goal_handle.succeed()
            result.success = True
            result.message = 'released, returned to watch'
        else:
            goal_handle.abort()
            result.success = False
            result.message = 'release_and_retry failed to return to watch'
        return result

    # ---- servo_pick ----

    def _servo_pick_tick(self):
        abort_reason = self.servo_loop.should_abort()
        if abort_reason is not None:
            self.get_logger().warn(f'servo_pick 중단 조건 발생: {abort_reason}')
            return ('ABORT', abort_reason)
        if self.servo_loop.should_close():
            if bool(self.get_parameter('debug.log_servo_decisions').value):
                self.get_logger().info('servo_pick 폐합 조건을 만족했습니다.')
            return ('CLOSE', None)
        return ('CONTINUE', None)

    def _servo_pick_tick_with_grasp_lock(self, goal_handle, request):
        """_servo_pick_tick()을 감싸 그리퍼 폐합을 x,y 추적과 병행시킨다.

        _servo_pick_tick()이 처음 'CLOSE'를 반환하는 순간에도 루프를 바로 끝내지
        않는다 - 대신 RG2 close()를 백그라운드 스레드로 시작만 해두고 'CONTINUE'를
        반환해 step()이 계속 돌게 한다. 그러면 그리퍼가 실제로 폐합되는 동안에도
        x,y 시각 서보 추적이 멈추지 않고(ServoLoop.step의 _grasp_locked로 z만 영구
        고정), 로봇팔이 마지막 순간의 속도값을 관성처럼 유지한 채 멈춰있지 않는다.
        스레드가 끝난 뒤에야 'CLOSE'를 반환해 루프를 종료한다.

        그리퍼를 실제로 닫기 "직전"에 취소/안전상태를 한 번 더 확인한다(기존
        _execute_servo_pick이 루프 종료 후 RG2 close 호출 전에 하던 것과 동일한
        안전판) - 그리퍼 폐합은 물리적으로 되돌리기 어려우므로, 이미 취소/FAULT가
        걸린 상태에서 새로 시작하지 않는다. 여기서 조용히 'CONTINUE'만 반환해도
        _run_rt_tracking의 루프 최상단이 다음 tick에서 그 취소/FAULT를 바로
        감지해 정상적으로 종료 경로를 타므로 따로 처리할 필요가 없다."""
        if self._servo_pick_close_thread is not None:
            if self._servo_pick_close_thread.is_alive():
                return ('CONTINUE', None)
            return ('CLOSE', None)
        state, reason = self._servo_pick_tick()
        if state != 'CLOSE':
            return (state, reason)
        if goal_handle.is_cancel_requested or self.safety_state != SafetyState.NORMAL:
            return ('CONTINUE', None)
        self._servo_pick_close_thread = threading.Thread(
            target=self._run_servo_pick_close, args=(goal_handle, request), daemon=True)
        self._servo_pick_close_thread.start()
        return ('CONTINUE', None)

    def _run_servo_pick_close(self, goal_handle, request):
        """백그라운드 스레드에서 실행 - RG2Client.close()는 그리퍼가 완전히 멈출
        때까지 블로킹하므로, 메인 RT 루프(_run_rt_tracking)가 그동안 x,y 추적을
        계속할 수 있도록 별도 스레드로 뺐다."""
        self._servo_pick_close_success = self.rg2_client.close(
            request.grasp_width_mm, request.grasp_force_n, goal_handle=goal_handle)

    def _servo_pick_step(self):
        """칼만 ServoLoop.step(tcp_pose, now)에 필요한 현재 TCP pose를 캐시에서
        읽어 넘긴다. 캐시가 아직 없거나 오래됐으면(_get_current_tcp_posx가 None을
        반환) 이번 틱은 명령을 계산하지 않고 건너뛴다 - 임의의 기본 좌표로
        제어식을 계산하지 않기 위함이다. _get_current_tcp_posx()는 posx 6-vector
        [x,y,z,A,B,C](mm/deg)이므로 위치(x,y,z)만 ServoLoop 단위(m)로 변환하고
        회전(A,B,C)은 deg 그대로 이어붙여 넘긴다 - yaw 제어가 tcp_pose[5](C각)를
        현재 손목 각도로 읽는다(ServoLoop.step 참고)."""
        tcp_pose_mm = self._get_current_tcp_posx()
        if tcp_pose_mm is None:
            self.get_logger().warn(
                'TCP 위치 캐시가 없거나 오래되어 이번 servo tick을 건너뜁니다.',
                throttle_duration_sec=1.0)
            return None
        tcp_pose = [value / 1000.0 for value in tcp_pose_mm[:3]] + list(tcp_pose_mm[3:6])
        command = self.servo_loop.step(tcp_pose, time.monotonic())
        if bool(self.get_parameter('debug.log_servo_decisions').value):
            snap = self.servo_loop.debug_snapshot()
            # 2026-07-11 실기: 락(vz=0) 이후에도 tcp z가 추가로 내려가는 현상의 원인
            # 후보 중 하나가 이 캐시의 staleness라 임시로 나이도 같이 남긴다 - throttle도
            # 0.5s->0.05s로 줄여 락 전후 수백 ms 구간을 촘촘히 볼 수 있게 한다. 진단
            # 끝나면 원래 값(0.5s, age_s 없이)으로 되돌려도 된다.
            cache_age_s = time.monotonic() - self._tcp_pose_cache['received_at']
            self.get_logger().info(
                f'servo_pick 속도 명령 계산: tcp_pose={tcp_pose} '
                f'z_gap={snap["last_z_gap_m"]} z_stable_count={snap["z_stable_count"]} '
                f'z_close_margin_ok={snap["z_close_margin_ok"]} '
                f'tool_speed_m_s={snap["tool_speed_m_s"]} v_stable_count={snap["v_stable_count"]} '
                f'depth_valid={snap["depth_valid"]} vz={snap["cmd_m_s"]["vz"]} '
                f'tcp_cache_age_s={cache_age_s:.3f} '
                f'e_xy_norm_m={snap["e_xy_norm_m"]} w={snap["w"]} '
                f'velocity_m_s={snap["velocity_m_s"]} '
                f'innovation_xy_m={snap["innovation_xy_m"]} '
                f'position_m={snap["position_m"]} '
                f'cmd_vx_vy={snap["cmd_m_s"]["vx"]},{snap["cmd_m_s"]["vy"]} '
                f'yaw_target_deg={snap["yaw_target_deg"]} yaw_error_deg={snap["yaw_error_deg"]} '
                f'yaw_stable_count={snap["yaw_stable_count"]} yaw_rate={snap["cmd_m_s"]["yaw_rate"]}',
                throttle_duration_sec=0.05)
        return command

    def _get_current_tcp_posx(self, max_age_parameter='servo_pick.tcp_pose_max_age_s'):
        """캐시된 최신 TCP 위치를 [x,y,z,rx,ry,rz](mm/deg)로 반환한다.

        실제 GetCurrentPosx 서비스 호출은 이 함수가 아니라 robot_control_node의
        _on_tf_broadcast_timer가 TF 방송과 함께 수행하고 결과를 self._tcp_pose_cache에
        저장한다(_tcp_tracking_active일 때만) - ToolTrack/HandTrack 콜백(60Hz일 수
        있음)마다 동기 서비스 호출을 하면 여러 요청이 겹쳐 executor 스레드를 점유해
        안전상태/E-Stop polling이 늦어질 수 있기 때문이다. 이 함수는 캐시만 읽으므로
        절대 블로킹하지 않는다. (2026-07-08: 이전에는 별도 _on_tcp_pose_refresh_timer가
        독립적으로 GetCurrentPosx를 폴링했으나, TF 방송 폴링과 같은 서비스를 이중
        호출해 스레드 고갈을 유발해 하나로 합쳤다.)

        hardware_enabled=false에서는 캐시가 애초에 채워지지 않으므로 항상 None을
        반환한다. 캐시가 없거나 max_age_parameter(서보 루프별로 다른 파라미터를 쓸 수
        있음 - servo_pick.tcp_pose_max_age_s / handover_servo.tcp_pose_max_age_s)보다
        오래됐으면(=조회 실패가 계속돼 오래된 값을 무한정 재사용하게 되는 경우 포함)
        None을 반환해 호출측이 임의의 기본 좌표를 쓰지 않게 한다."""
        if not self.hardware_enabled:
            return None
        cache = self._tcp_pose_cache
        if cache is None:
            return None
        max_age_s = self.get_parameter(max_age_parameter).value
        age_s = time.monotonic() - cache['received_at']
        if age_s > max_age_s:
            return None
        return cache['pos6']

    def _validate_tool_track_message(self, message) -> bool:
        """servo_loop(칼만 ServoLoop)는 msg.pose.position을 base_link 기준 절대
        목표 위치로 직접 필터에 흘려보내므로(TCP 오차로 변환하지 않음), 여기서는
        frame_id/NaN만 확인한다."""
        expected_frame = self.get_parameter('servo_pick.tool_track_frame_id').value
        if message.header.frame_id != expected_frame:
            self.get_logger().error(
                f"frame_id='{message.header.frame_id}'가 '{expected_frame}'가 아닙니다.")
            return False
        position = message.pose.position
        valid = all(math.isfinite(value) for value in (
            position.x, position.y, position.z))
        if not valid:
            self.get_logger().warn(
                f'ToolTrack 좌표가 NaN/Inf입니다: ({position.x}, {position.y}, {position.z})',
                throttle_duration_sec=1.0)
        return valid

    def _validate_servo_command(self, cmd) -> bool:
        """속도 명령을 실제로 발행하기 직전 마지막 안전 검사. ServoLoop.step()이
        내부적으로 이미 _clip으로 속도를 제한하지만, 발행 경계에서 NaN/Inf와 제한을
        한 번 더 확인해 유효하지 않은 값이 그대로 나가지 않게 한다."""
        values = (cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate)
        if not all(math.isfinite(v) for v in values):
            self.get_logger().error(f'서보 속도 명령에 NaN/Inf가 포함되어 있습니다: {values}')
            return False
        tol = self.get_parameter('servo.command_validate_tolerance').value
        v_max = abs(self.get_parameter('servo.v_max').value)
        descend_speed = abs(self.get_parameter('servo.descend_speed').value)
        # yaw_rate는 deg/s, v_max/vx/vy는 m/s - 단위가 다르므로 별도 파라미터로
        # 검증한다(이전에는 v_max를 그대로 재사용해 yaw_rate가 사실상 항상 0으로
        # 고정됐던 동안 드러나지 않은 버그였다).
        yaw_rate_max = abs(self.get_parameter('servo.yaw_rate_max_deg_s').value)
        if abs(cmd.vx) > v_max + tol or abs(cmd.vy) > v_max + tol:
            self.get_logger().error(
                f'서보 속도 명령이 v_max({v_max}) 제한을 넘었습니다: vx={cmd.vx}, vy={cmd.vy}')
            return False
        if abs(cmd.yaw_rate) > yaw_rate_max + tol:
            self.get_logger().error(
                f'서보 yaw_rate 명령이 yaw_rate_max_deg_s({yaw_rate_max}) 제한을 '
                f'넘었습니다: yaw_rate={cmd.yaw_rate}')
            return False
        if abs(cmd.vz) > descend_speed + tol:
            self.get_logger().error(
                f'서보 z 속도 명령이 descend_speed({descend_speed}) 제한을 넘었습니다: vz={cmd.vz}')
            return False
        return True

    def _validate_lift_command(self, cmd) -> bool:
        """_run_grasp_lift 전용 발행 경계 검사(_validate_servo_command와 동일 형태) -
        들어올림은 servo_pick.lift_speed_m_s를 쓰는데 이 값이 servo.descend_speed보다
        커도 되므로(위쪽엔 장애물이 없어 표면 근접 제동거리 제약을 받지 않음),
        descend_speed 기준인 _validate_servo_command를 그대로 쓰면 항상 거부된다."""
        values = (cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate)
        if not all(math.isfinite(v) for v in values):
            self.get_logger().error(f'들어올림 속도 명령에 NaN/Inf가 포함되어 있습니다: {values}')
            return False
        tol = self.get_parameter('servo.command_validate_tolerance').value
        lift_speed_m_s = abs(self.get_parameter('servo_pick.lift_speed_m_s').value)
        yaw_rate_max = abs(self.get_parameter('servo.yaw_rate_max_deg_s').value)
        if abs(cmd.vz) > lift_speed_m_s + tol:
            self.get_logger().error(
                f'들어올림 속도 명령이 lift_speed_m_s({lift_speed_m_s}) 제한을 넘었습니다: vz={cmd.vz}')
            return False
        if abs(cmd.yaw_rate) > yaw_rate_max + tol:
            self.get_logger().error(
                f'들어올림 yaw_rate 명령이 yaw_rate_max_deg_s({yaw_rate_max}) 제한을 '
                f'넘었습니다: yaw_rate={cmd.yaw_rate}')
            return False
        return True

    def _on_tool_track_during_servo(self, msg):
        if not self._validate_tool_track_message(msg):
            # frame_id 불일치 또는 NaN/Inf - 이번 프레임은 유실된 것처럼 취급한다
            # (ServoLoop.should_abort의 t_lost_s가 결국 감지한다).
            return
        self.servo_loop.on_tool_track(msg)

    def _run_rt_tracking(
            self, goal_handle, *, name, message_type, topic, callback,
            servo, step, tick, validate_command, ready_parameter,
            period_parameter, accel_param_prefix):
        """물체·손 추적이 공통으로 사용하는 실행/취소/정리 루프.

        RT 세션 없이 speedl_stream(비-RT)에 직접 연속 발행한다(2026-07-07
        probe_speedl_stream.py 실측: RT 세션 없이도 부드러운 서보잉이 가능함을
        확인). RT가 제공하던 "명령 끊기면 자동 정지"는 SpeedlWatchdog(데드맨
        스위치, 별도 스레드)로 대체한다 - 루프가 매 틱 pet()하고,
        watchdog_timeout_s 이내에 pet이 없으면 워치독이 독립적으로 vel=0을
        발행한다(단일 정지 명령으로 충분함도 같은 실측으로 확인됨).

        step: 인자 없이 호출해 이번 틱의 ServoCommand(또는 아직 계산할 수 없으면
        None)를 반환하는 콜러블 - servo_pick(칼만 ServoLoop)과 handover_approach
        (HandServoLoop) 둘 다 이 루프를 쓴다."""
        subscription = None
        outcome = 'ABORT'
        detail = f'{name} aborted'
        self._tcp_tracking_active = True
        watchdog = None
        try:
            watchdog = SpeedlWatchdog(
                timeout_s=float(
                    self.get_parameter(f'{accel_param_prefix}.watchdog_timeout_s').value),
                on_timeout=lambda: self._doosan.publish_speedl(
                    ServoCommand(), accel_param_prefix=accel_param_prefix,
                    period_param_name=period_parameter),
                poll_interval_s=float(
                    self.get_parameter(f'{accel_param_prefix}.watchdog_poll_interval_s').value))
            subscription = self.create_subscription(
                message_type, topic, callback, 10,
                callback_group=self.sensor_callback_group)
            ready = bool(self.get_parameter(ready_parameter).value)
            period = float(self.get_parameter(period_parameter).value)
            publish_active = self.hardware_enabled and ready and self._doosan is not None
            if publish_active:
                watchdog.start()

            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    recoverable_stop_mode = self.get_parameter(
                        'safety.recoverable_stop_mode').value
                    if not self._cleanup_stop_motion(recoverable_stop_mode):
                        detail = f'{name} 취소 중 MoveStop 실패'
                        self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                        outcome = 'FAULT'
                    else:
                        outcome, detail = 'CANCELED', f'{name} canceled'
                    break
                if self.safety_state != SafetyState.NORMAL:
                    outcome = 'ABORT'
                    detail = f'{name} aborted - safety_state={self.safety_state}'
                    break

                state, reason = tick()
                feedback = RobotTask.Feedback()
                feedback.state = servo.get_state()
                goal_handle.publish_feedback(feedback)
                if state == 'ABORT':
                    # 취소/예외 경로와 달리 이 분기가 그냥 break만 하던 시절에는
                    # SpeedlWatchdog이 곧장 stop()으로 해제되어(finally 블록)
                    # 타임아웃이 뜰 새도 없이 로봇이 마지막 속도로 계속 움직였다
                    # (2026-07-10 실기에서 diverging abort 후 관성 하강으로 바닥 충돌
                    # 확인). 취소 경로와 동일하게 여기서도 명시적으로 MoveStop을 건다.
                    recoverable_stop_mode = self.get_parameter(
                        'safety.recoverable_stop_mode').value
                    if not self._cleanup_stop_motion(recoverable_stop_mode):
                        detail = f'{name} aborted({reason}) 중 MoveStop 실패'
                        self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                        outcome = 'FAULT'
                    else:
                        outcome, detail = 'ABORT', reason
                    break
                if state in ('CLOSE', 'STOP'):
                    outcome, detail = 'ARRIVED', ''
                    break

                command = step()
                if command is not None and publish_active:
                    if not validate_command(command):
                        outcome = 'ABORT'
                        detail = f'{name} aborted - invalid velocity command'
                        break
                    self._doosan.publish_speedl(
                        command, accel_param_prefix=accel_param_prefix,
                        period_param_name=period_parameter)
                    watchdog.pet()
                time.sleep(period)
            else:
                self._cleanup_stop_motion()
                outcome, detail = 'ABORT', f'{name} aborted - rclpy 종료 중'
        except Exception as exc:
            self.get_logger().error(f'{name} 실행 중 예외: {exc}')
            stop_ok = self._cleanup_stop_motion()
            if goal_handle.is_cancel_requested and stop_ok:
                outcome, detail = 'CANCELED', f'{name} canceled after exception: {exc}'
            else:
                outcome, detail = 'ABORT', f'{name} exception: {exc}'
                if not stop_ok:
                    self._declare_fault(
                        f'{FaultPrefix.FAULT}{name} 예외 처리 중 MoveStop 실패: {exc}')
                    outcome = 'FAULT'
        finally:
            if watchdog is not None:
                watchdog.stop()
            sub_ok = self._cleanup_destroy_subscription(subscription)
            self._tcp_tracking_active = False
            if not sub_ok:
                detail = f'{name} cleanup 실패 (subscription={sub_ok})'
                self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                outcome = 'FAULT'
        return outcome, detail

    @staticmethod
    def _finish_tracking_result(goal_handle, outcome, detail):
        result = RobotTask.Result()
        result.success = outcome == 'ARRIVED'
        result.message = detail
        if outcome == 'ARRIVED':
            goal_handle.succeed()
        elif outcome == 'CANCELED':
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return result

    def _run_grasp_lift(self, goal_handle):
        """그리퍼 폐합 확인 직후, VERIFY_GRASP 판정 전에 z를 servo_pick.lift_height_m
        만큼 들어올린다(docs/전체 계획.md 2/9번 "즉시 들어올림") - 검증 결과와
        무관하게 항상 수행하며, 검증 실패로 release_and_retry가 이어지더라도
        원래 높이로 되돌리지 않고 이 들어올려진 위치에서 그대로 연다(2026-07-12
        확정). xy는 그리퍼가 이미 공구를 붙잡고 있어 더 추적할 필요가 없으므로
        vz만 명령한다. 속도는 servo_pick.lift_speed_m_s를 쓴다 - 위쪽엔 장애물이
        없어 표면 근접 제동거리로 신중하게 잡은 servo.descend_speed보다 빠르게
        둘 수 있다(2026-07-12 확정)."""
        height_m = float(self.get_parameter('servo_pick.lift_height_m').value)
        if height_m <= 0.0:
            return 'ARRIVED', ''
        publish_active = (
            self.hardware_enabled and self._doosan is not None
            and bool(self.get_parameter('servo_pick.hardware_ready').value))
        if not publish_active:
            return 'ARRIVED', ''

        # _run_rt_tracking이 xy 추적 종료 직후 이 플래그를 이미 False로 되돌려놔서
        # (_tf_broadcast_timer가 TCP 위치 캐시 갱신을 멈춘 상태, _on_tf_broadcast_timer
        # 참고) - 들어올림 동안 z 진행을 확인하려면 다시 켜둬야 한다. 이 함수를 나가는
        # 모든 경로(ARRIVED/CANCELED/ABORT/FAULT/예외)에서 반드시 원래대로 꺼야
        # 하므로 아래 try/finally로 감싼다.
        self._tcp_tracking_active = True
        try:
            return self._run_grasp_lift_loop(goal_handle, height_m)
        finally:
            self._tcp_tracking_active = False

    def _run_grasp_lift_loop(self, goal_handle, height_m):
        start_tcp_mm = self._get_current_tcp_posx()
        if start_tcp_mm is None:
            detail = 'servo_pick 들어올림 시작 실패 - TCP 위치 캐시 없음'
            self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
            return 'FAULT', detail
        start_z_m = start_tcp_mm[2] / 1000.0
        speed_m_s = abs(float(self.get_parameter('servo_pick.lift_speed_m_s').value))
        period = float(self.get_parameter('servo_pick.control_period_s').value)
        deadline = time.monotonic() + float(self.get_parameter('servo_pick.lift_timeout_s').value)

        watchdog = SpeedlWatchdog(
            timeout_s=float(self.get_parameter('servo_pick.watchdog_timeout_s').value),
            on_timeout=lambda: self._doosan.publish_speedl(
                ServoCommand(), accel_param_prefix='servo_pick',
                period_param_name='servo_pick.control_period_s'),
            poll_interval_s=float(
                self.get_parameter('servo_pick.watchdog_poll_interval_s').value))
        watchdog.start()
        outcome, detail = 'ARRIVED', ''
        try:
            while rclpy.ok():
                feedback = RobotTask.Feedback()
                feedback.state = ServoState.LIFTING
                goal_handle.publish_feedback(feedback)
                if goal_handle.is_cancel_requested:
                    recoverable_stop_mode = self.get_parameter(
                        'safety.recoverable_stop_mode').value
                    if not self._cleanup_stop_motion(recoverable_stop_mode):
                        detail = 'servo_pick 들어올림 취소 중 MoveStop 실패'
                        self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                        outcome = 'FAULT'
                    else:
                        outcome, detail = 'CANCELED', 'servo_pick 들어올림 canceled'
                    break
                if self.safety_state != SafetyState.NORMAL:
                    recoverable_stop_mode = self.get_parameter(
                        'safety.recoverable_stop_mode').value
                    self._cleanup_stop_motion(recoverable_stop_mode)
                    outcome, detail = (
                        'ABORT', f'servo_pick 들어올림 aborted - safety_state={self.safety_state}')
                    break

                if time.monotonic() >= deadline:
                    recoverable_stop_mode = self.get_parameter(
                        'safety.recoverable_stop_mode').value
                    self._cleanup_stop_motion(recoverable_stop_mode)
                    detail = 'servo_pick 들어올림 타임아웃 - 목표 높이에 도달하지 못했습니다'
                    self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                    outcome = 'FAULT'
                    break

                tcp_mm = self._get_current_tcp_posx()
                if tcp_mm is None:
                    time.sleep(period)
                    continue
                if (tcp_mm[2] / 1000.0) - start_z_m >= height_m:
                    break

                command = ServoCommand(vz=speed_m_s)
                if not self._validate_lift_command(command):
                    recoverable_stop_mode = self.get_parameter(
                        'safety.recoverable_stop_mode').value
                    self._cleanup_stop_motion(recoverable_stop_mode)
                    outcome, detail = 'ABORT', 'servo_pick 들어올림 aborted - invalid velocity command'
                    break
                self._doosan.publish_speedl(
                    command, accel_param_prefix='servo_pick',
                    period_param_name='servo_pick.control_period_s')
                watchdog.pet()
                time.sleep(period)
            else:
                self._cleanup_stop_motion()
                outcome, detail = 'ABORT', 'servo_pick 들어올림 aborted - rclpy 종료 중'
        except Exception as exc:
            self.get_logger().error(f'servo_pick 들어올림 중 예외: {exc}')
            stop_ok = self._cleanup_stop_motion()
            if goal_handle.is_cancel_requested and stop_ok:
                outcome, detail = 'CANCELED', f'servo_pick 들어올림 canceled after exception: {exc}'
            else:
                outcome, detail = 'ABORT', f'servo_pick 들어올림 exception: {exc}'
                if not stop_ok:
                    self._declare_fault(
                        f'{FaultPrefix.FAULT}servo_pick 들어올림 예외 처리 중 MoveStop 실패: {exc}')
                    outcome = 'FAULT'
        finally:
            watchdog.stop()
        return outcome, detail

    def _execute_servo_pick(self, goal_handle):
        if self.safety_state != SafetyState.NORMAL:
            return self._finish_tracking_result(
                goal_handle, 'ABORT',
                f'servo_pick rejected - safety_state={self.safety_state}')
        if (self.hardware_enabled
                and not self.get_parameter('servo_pick.hardware_ready').value):
            return self._finish_tracking_result(
                goal_handle, 'ABORT',
                'servo_pick rejected - servo_pick.hardware_ready=false')

        request = goal_handle.request
        self.servo_loop.start(
            request.tool_class, request.grasp_width_mm, request.grasp_force_n)
        # should_close() 만족 시 _servo_pick_tick_with_grasp_lock이 여기 채워 넣는다
        # (RG2 close를 백그라운드로 돌리는 동안에도 x,y 추적을 계속하기 위함 - 그
        # 함수 독스트링 참고). 매 goal마다 새로 초기화해 이전 goal의 잔여 상태가
        # 새지 않게 한다.
        self._servo_pick_close_thread = None
        self._servo_pick_close_success = None
        outcome, detail = self._run_rt_tracking(
            goal_handle,
            name='servo_pick',
            message_type=ToolTrack,
            topic='/vision/tool_track',
            callback=self._on_tool_track_during_servo,
            servo=self.servo_loop,
            step=self._servo_pick_step,
            tick=lambda: self._servo_pick_tick_with_grasp_lock(goal_handle, request),
            validate_command=self._validate_servo_command,
            ready_parameter='servo_pick.hardware_ready',
            period_parameter='servo_pick.control_period_s',
            accel_param_prefix='servo_pick')
        # _run_rt_tracking이 어떤 경로(정상 CLOSE/취소/중단/예외)로 빠져나왔든, RG2
        # close 스레드가 떠 있는 채로 이 함수를 반환하면 안 되므로 항상 join한다.
        # 스레드 내부의 rg2_client.close()도 goal_handle.is_cancel_requested/
        # safety_state를 스스로 감시해 필요시 그리퍼에 stop을 보내고 끝나므로
        # join이 무한정 걸리지는 않는다.
        if self._servo_pick_close_thread is not None:
            self._servo_pick_close_thread.join()

        if outcome != 'ARRIVED':
            return self._finish_tracking_result(goal_handle, outcome, detail)

        # 이 지점에 도달했다는 건 tick()이 'CLOSE'를 반환했다는 뜻 = RG2 close
        # 스레드가 이미 완료됐다는 뜻이므로(위 join은 안전을 위한 재확인일 뿐),
        # 별도로 close()를 다시 호출하지 않고 스레드가 남긴 결과를 그대로 쓴다.
        if not self._servo_pick_close_success:
            if self.rg2_client.last_status == RG2Status.CANCELED:
                self._cleanup_stop_motion()
                return self._finish_tracking_result(
                    goal_handle, 'CANCELED', 'servo_pick canceled during RG2 close')
            detail = f'servo_pick RG2 close 실패(status={self.rg2_client.last_status})'
            self._checkpoint_event('D', 'gripper_closed', 'FAIL', detail)
            self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
            return self._finish_tracking_result(goal_handle, 'FAULT', detail)

        self._checkpoint_event(
            'D', 'gripper_closed', 'PASS', '그리퍼가 정상적으로 닫혔습니다.',
            {'grasp_width_mm': request.grasp_width_mm, 'grasp_force_n': request.grasp_force_n})
        width_mm, grip_detected = self.rg2_client.get_state()
        # 그리퍼가 완전히 닫히고 grip_detected를 확인한 지금에서야 x,y 추적도 함께
        # 정지시킨다 (vz는 그리퍼가 닫히는 내내 _grasp_locked로 이미 0에 고정돼 있었다).
        self._cleanup_stop_motion()
        lift_outcome, lift_detail = self._run_grasp_lift(goal_handle)
        if lift_outcome != 'ARRIVED':
            return self._finish_tracking_result(goal_handle, lift_outcome, lift_detail)
        # 들어올림이 실제 속도(lift_speed_m_s)로 움직이던 중이라, 인자 없이 부르면
        # fault_stop_mode(QUICK/Cat.2, 실제 안전 FAULT 전용)로 급정지되어 외력
        # 스파이크로 오인되어 FAULT가 뜬다(2026-07-12 실기 확인) - 정상 종료이므로
        # recoverable_stop_mode를 명시한다.
        self._cleanup_stop_motion(self.get_parameter('safety.recoverable_stop_mode').value)
        result = self._finish_tracking_result(goal_handle, 'ARRIVED', '')
        result.final_width_mm = width_mm
        result.grip_detected = grip_detected
        return result

    # ---- handover_approach (handover_safe 이후 작업자 손에 연속 접근) ----

    def _validate_hand_track_message(self, message) -> bool:
        """HandServoLoop는 msg.pose.position을 base_link 기준 절대 손 위치로 직접
        사용하므로(TCP 오차로 변환하지 않음), 여기서는 frame_id/NaN만 확인한다
        (_validate_tool_track_message와 동일 형태)."""
        expected_frame = self.get_parameter('handover_servo.hand_track_frame_id').value
        if message.header.frame_id != expected_frame:
            self.get_logger().error(
                f"frame_id='{message.header.frame_id}'가 '{expected_frame}'가 아닙니다.")
            return False
        position = message.pose.position
        valid = all(math.isfinite(value) for value in (
            position.x, position.y, position.z))
        if not valid:
            self.get_logger().warn(
                f'HandTrack 좌표가 NaN/Inf입니다: ({position.x}, {position.y}, {position.z})',
                throttle_duration_sec=1.0)
        return valid

    def _validate_handover_servo_command(self, cmd) -> bool:
        """속도 명령을 실제로 발행하기 직전 마지막 안전 검사(_validate_servo_command와
        동일 형태). HandServoLoop는 x/y/z를 모두 같은 v_max로 클립하므로(3D P 제어,
        servo_pick처럼 descend_speed로 z를 별도 다루지 않음) 세 축 모두 v_max 하나로
        검사한다."""
        values = (cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate)
        if not all(math.isfinite(v) for v in values):
            self.get_logger().error(f'서보 속도 명령에 NaN/Inf가 포함되어 있습니다: {values}')
            return False
        tol = self.get_parameter('handover_servo.command_validate_tolerance').value
        v_max = abs(self.get_parameter('handover_servo.v_max').value)
        if (abs(cmd.vx) > v_max + tol or abs(cmd.vy) > v_max + tol
                or abs(cmd.vz) > v_max + tol or abs(cmd.yaw_rate) > v_max + tol):
            self.get_logger().error(
                f'서보 속도 명령이 v_max({v_max}) 제한을 넘었습니다: '
                f'vx={cmd.vx}, vy={cmd.vy}, vz={cmd.vz}, yaw_rate={cmd.yaw_rate}')
            return False
        return True

    def _on_hand_track_during_servo(self, msg):
        if not self._validate_hand_track_message(msg):
            # frame_id 불일치 또는 NaN/Inf - 이번 프레임은 유실된 것처럼 취급한다
            # (HandServoLoop.tick의 t_lost_s가 결국 감지한다).
            return
        self.hand_servo_loop.on_hand_track(msg)

    def _handover_approach_step(self):
        """_servo_pick_step과 동일 형태 - 캐시된 TCP 위치를 HandServoLoop.step에 넘긴다."""
        tcp_pose_mm = self._get_current_tcp_posx('handover_servo.tcp_pose_max_age_s')
        if tcp_pose_mm is None:
            self.get_logger().warn(
                'TCP 위치 캐시가 없거나 오래되어 이번 handover_approach tick을 건너뜁니다.',
                throttle_duration_sec=1.0)
            return None
        tcp_pose_m = [value / 1000.0 for value in tcp_pose_mm[:3]]
        command = self.hand_servo_loop.step(tcp_pose_m, time.monotonic())
        if bool(self.get_parameter('debug.log_servo_decisions').value):
            self.get_logger().info(
                f'handover_approach 속도 명령 계산: tcp_pose_m={tcp_pose_m}',
                throttle_duration_sec=0.5)
        return command

    def _handover_approach_tick(self):
        state, reason = self.hand_servo_loop.tick()
        if state == 'ABORT':
            self.get_logger().warn(f'handover_approach 중단 조건 발생: {reason}')
        elif state == 'STOP':
            if bool(self.get_parameter('debug.log_servo_decisions').value):
                self.get_logger().info('handover_approach가 주먹 확정으로 정지합니다.')
        return state, reason

    def _execute_handover_approach(self, goal_handle):
        if self.safety_state != SafetyState.NORMAL:
            return self._finish_tracking_result(
                goal_handle, 'ABORT',
                f'handover_approach rejected - safety_state={self.safety_state}')
        if (self.hardware_enabled
                and not self.get_parameter('handover_servo.hardware_ready').value):
            return self._finish_tracking_result(
                goal_handle, 'ABORT',
                'handover_approach rejected - handover_servo.hardware_ready=false')

        if not self.hardware_enabled:
            # dry-run은 다른 _execute_*와 동일하게 센서/하드웨어 의존 없이 흐름만
            # 검증한다 - hand_track이 아직 vision_node에서 발행되지 않는 환경에서도
            # 실행 경로(취소/성공)를 테스트할 수 있다.
            if self._dry_run_move(goal_handle):
                return self._finish_tracking_result(goal_handle, 'ARRIVED', '')
            if goal_handle.is_cancel_requested:
                return self._finish_tracking_result(
                    goal_handle, 'CANCELED', 'handover_approach canceled (dry_run)')
            return self._finish_tracking_result(
                goal_handle, 'ABORT',
                f'handover_approach aborted (dry_run) - safety_state={self.safety_state}')

        self.hand_servo_loop.start()
        outcome, detail = self._run_rt_tracking(
            goal_handle,
            name='handover_approach',
            message_type=HandTrack,
            topic='/vision/hand_track',
            callback=self._on_hand_track_during_servo,
            servo=self.hand_servo_loop,
            step=self._handover_approach_step,
            tick=self._handover_approach_tick,
            validate_command=self._validate_handover_servo_command,
            ready_parameter='handover_servo.hardware_ready',
            period_parameter='handover_servo.control_period_s',
            accel_param_prefix='handover_servo')
        if outcome == 'ARRIVED':
            # 주먹 확정으로 도착한 경우(긴급정지가 아니라 의도된 정지) - _run_rt_tracking은
            # tick()이 'STOP'을 반환하면 그 자리에서 break만 하고 실제 정지 명령은 보내지
            # 않는다(SpeedlWatchdog은 이 시점엔 이미 watchdog.stop()으로 해제되어 타임아웃이
            # 발동하지 않는다). servo_pick의 CLOSE와 달리 이 STOP은 속도가 이미 0에
            # 수렴했다는 보장이 없으므로(should_close 같은 수렴 조건이 없음), 여기서
            # 명시적으로 MoveStop을 걸어 로봇을 확실히 멈춘다. 실패하면 로봇이 계속 움직이고
            # 있을 수 있으므로 FAULT로 승격한다 - safety_state를 바꾸는 것이지 이 정지 자체가
            # 긴급정지라는 뜻은 아니다(뒤이어 task_manager가 handover_hold로 정상 진행한다).
            if not self._cleanup_stop_motion():
                detail = 'handover_approach 정지(MoveStop) 실패'
                self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                return self._finish_tracking_result(goal_handle, 'FAULT', detail)
        return self._finish_tracking_result(goal_handle, outcome, detail)

    # ---- handover_hold ----

    def _enable_compliance(self) -> None:
        if not self.hardware_enabled:
            self.get_logger().info('[dry_run] compliance 모드 on 생략')
            return
        if self._doosan is None:
            raise RuntimeError('DoosanDriver가 초기화되지 않았습니다.')
        self._doosan.enable_compliance()

    def _disable_compliance(self) -> None:
        if not self.hardware_enabled:
            self.get_logger().info('[dry_run] compliance 모드 off 생략')
            return
        if self._doosan is None:
            return
        self._doosan.disable_compliance()

    def _is_pull_detected(self, robot_state) -> bool:
        """robot_state의 tool_force에서 전달 방향(pull_axis_index) 성분만 확인해
        판정한다. 다른 축의 힘/토크는 무시하므로, handover_hold 중 임의 방향의
        접촉을 당김으로 오판하지 않는다 (요구사항: 전달 방향의 당김만 정상 전달).

        pull_axis_index는 tool_force의 힘 성분(0=x,1=y,2=z)만 허용한다. 3~5(모멘트,
        Nm)는 힘 임계값(pull_force_threshold_n, N)과 단위가 달라 비교 대상이 아니므로
        허용하지 않는다.

        TODO: pull_axis_index/pull_direction_sign/pull_force_threshold_n은 실제
        그리퍼-TCP 장착 방향과 전달 자세에 따라 달라지는 캘리브레이션 값이다.
        하드웨어 셋업 전에는 임의로 축을 추측하지 않기 위해 기본값을 -1(미설정)로
        두었고, 이 경우 항상 False를 반환해 오탐(당김 오판)을 방지한다.
        """
        if not isinstance(robot_state, dict):
            return False
        axis = int(self.get_parameter('handover_hold.pull_axis_index').value)
        if axis < 0 or axis > 2:
            # handover_hold.poll_interval_s(기본 0.05s)마다 호출되므로 throttle
            # 없이는 로그가 초당 수십 줄씩 쏟아진다.
            self.get_logger().warn(
                'handover_hold.pull_axis_index 미설정(또는 모멘트 축 지정) - '
                '당김 감지를 비활성화합니다 (힘 성분 0,1,2만 허용).',
                throttle_duration_sec=1.0)
            return False
        tool_force = robot_state.get('tool_force')
        if not tool_force:
            return False
        sign = self.get_parameter('handover_hold.pull_direction_sign').value
        threshold_n = self.get_parameter('handover_hold.pull_force_threshold_n').value
        component = sign * tool_force[axis]
        return component > threshold_n

    @staticmethod
    def _is_fresh_robot_state(robot_state, since_monotonic: float, max_age_s: float) -> bool:
        """robot_state 샘플이 since_monotonic(예: handover_hold 시작 시각) 이후에
        수신됐고, 수신된 지 max_age_s 이내인 경우에만 신선하다고 판단한다.
        handover_hold 시작 전의 오래된 샘플을 당김 판정에 사용하지 않기 위함이다."""
        if not isinstance(robot_state, dict):
            return False
        received_at = robot_state.get('received_at')
        if received_at is None or received_at < since_monotonic:
            return False
        return (time.monotonic() - received_at) <= max_age_s

    def _execute_handover_hold(self, goal_handle):
        if self.safety_state != SafetyState.NORMAL:
            return self._finish_tracking_result(
                goal_handle, 'ABORT',
                f'handover_hold rejected - safety_state={self.safety_state}')

        compliance_on = False
        try:
            self._enable_compliance()
            compliance_on = True
            # 이 구간은 사람의 접촉력이 기대되는 상태(당김 감지는 아래 wait_for_pull이
            # 전담)이므로, 관절 토크 기준 범용 충돌 FAULT(DrflForceMonitor)는 일시
            # 정지한다 - 안 그러면 당김/접촉이 두산 관절 토크 임계값을 먼저 넘겨
            # wait_for_pull이 확정되기 전에 FAULT로 중단되어 버린다.
            self._suspend_drfl_force_monitor()
            self._checkpoint_event(
                'I', 'compliance_mode_active', 'PASS', '컴플라이언스 모드가 가동되었습니다.')
            outcome = self.safety_monitor.wait_for_pull(
                goal_handle, self._is_pull_detected, self._is_fresh_robot_state)
            if outcome == 'CANCELED':
                stop_ok = self._cleanup_stop_motion()
                compliance_ok = self._cleanup_disable_compliance()
                compliance_on = False
                if stop_ok and compliance_ok:
                    return self._finish_tracking_result(
                        goal_handle, 'CANCELED', 'handover_hold canceled')
                detail = (
                    'handover_hold 취소 cleanup 실패 '
                    f'(move_stop={stop_ok}, compliance={compliance_ok})')
                self._checkpoint_event('I', 'compliance_mode_ended', 'FAIL', detail)
                self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                return self._finish_tracking_result(goal_handle, 'FAULT', detail)
            if outcome != 'PULLED':
                return self._finish_tracking_result(
                    goal_handle, 'ABORT', f'handover_hold {outcome.lower()}')

            compliance_ok = self._cleanup_disable_compliance()
            compliance_on = False
            if not compliance_ok:
                detail = 'handover_hold compliance 해제 실패 - RG2를 열지 않습니다.'
                self._checkpoint_event('I', 'compliance_mode_ended', 'FAIL', detail)
                self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                return self._finish_tracking_result(goal_handle, 'FAULT', detail)
            self._checkpoint_event(
                'I', 'compliance_mode_ended', 'PASS', '컴플라이언스 모드가 종료되었습니다.')
            if goal_handle.is_cancel_requested:
                self._cleanup_stop_motion()
                return self._finish_tracking_result(
                    goal_handle, 'CANCELED',
                    'handover_hold canceled before RG2 open')
            if self.safety_state != SafetyState.NORMAL:
                return self._finish_tracking_result(
                    goal_handle, 'ABORT',
                    f'handover_hold aborted - safety_state={self.safety_state}')

            if not self.rg2_client.open(goal_handle=goal_handle):
                if self.rg2_client.last_status == RG2Status.CANCELED:
                    self._cleanup_stop_motion()
                    return self._finish_tracking_result(
                        goal_handle, 'CANCELED',
                        'handover_hold canceled during RG2 open')
                detail = f'handover_hold RG2 open 실패(status={self.rg2_client.last_status})'
                self._checkpoint_event('I', 'gripper_opened_on_pull', 'FAIL', detail)
                self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                return self._finish_tracking_result(goal_handle, 'FAULT', detail)
            self._checkpoint_event(
                'I', 'gripper_opened_on_pull', 'PASS', '당김 감지 후 그리퍼가 개방되었습니다.')
            return self._finish_tracking_result(
                goal_handle, 'ARRIVED', 'pull_detected, released')
        except Exception as exc:
            self.get_logger().error(f'handover_hold 실행 중 예외: {exc}')
            self._cleanup_stop_motion()
            outcome = 'CANCELED' if goal_handle.is_cancel_requested else 'ABORT'
            return self._finish_tracking_result(
                goal_handle, outcome, f'handover_hold exception: {exc}')
        finally:
            self._resume_drfl_force_monitor()
            if compliance_on:
                self._cleanup_disable_compliance()
