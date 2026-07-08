import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped

from handover_interfaces.action import RobotTask
from handover_interfaces.msg import ToolTrack

from robot_control.rg2_client import RG2Status
from robot_control.safety_monitor import FaultPrefix, SafetyState
from robot_control.servo_loop import ServoCommand
from robot_control.speedl_watchdog import SpeedlWatchdog


class TaskExecutor:
    """RobotTask 5종을 실행하는 RobotControlNode용 mixin."""

    def _cleanup_stop_motion(self) -> bool:
        """M0609 MoveStop 정지를 시도한다. 실패/예외 시에도 절대 상위로 전파하지 않는다."""
        if not self.hardware_enabled or self._doosan is None:
            return True
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
            if named_target == 'home':
                # 홈으로 이동하기 전에는 다음 작업을 위해 그리퍼가 열린 상태임을
                # 보장한다 - 이전에 어떤 경로로 왔든(수동 이동, pick 중단, 검증
                # 실패 등) 그리퍼가 물체를 쥔 채로 홈에 도착하지 않도록 하는
                # 안전/리셋 동작이다(_execute_release_and_retry의 RG2 open 처리와
                # 동일한 패턴). 이미 열려있으면 불필요한 재통신을 피해 생략한다.
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = 'move_named(home) canceled before opening gripper'
                    return result
                if self.safety_state != SafetyState.NORMAL:
                    goal_handle.abort()
                    result.success = False
                    result.message = (
                        f'move_named(home) aborted before opening gripper - '
                        f'safety_state={self.safety_state}')
                    return result
                if not self._is_gripper_already_open():
                    if not self.rg2_client.open(goal_handle=goal_handle):
                        if self.rg2_client.last_status == RG2Status.CANCELED:
                            goal_handle.canceled()
                            result.success = False
                            result.message = 'move_named(home) canceled during RG2 open'
                            return result
                        detail = (
                            f'move_named(home) RG2 open 실패'
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
            return ('ABORT', abort_reason)
        if self.servo_loop.should_close():
            return ('CLOSE', None)
        return ('CONTINUE', None)

    def _servo_pick_step(self):
        """칼만 ServoLoop.step(tcp_pose, now)에 필요한 현재 TCP 위치를 캐시에서
        읽어 넘긴다. 캐시가 아직 없거나 오래됐으면(_get_current_tcp_posx가 None을
        반환) 이번 틱은 명령을 계산하지 않고 건너뛴다 - 임의의 기본 좌표로
        제어식을 계산하지 않기 위함이다. _get_current_tcp_posx()는 mm 단위이므로
        ServoLoop(m 단위, ToolTrack과 동일)에 맞게 변환한다."""
        tcp_pose_mm = self._get_current_tcp_posx()
        if tcp_pose_mm is None:
            return None
        tcp_pose_m = [value / 1000.0 for value in tcp_pose_mm[:3]]
        return self.servo_loop.step(tcp_pose_m, time.monotonic())

    def _get_current_tcp_posx(self):
        """캐시된 최신 TCP 위치를 [x,y,z,rx,ry,rz](mm/deg)로 반환한다.

        실제 GetCurrentPosx 서비스 호출은 이 함수가 아니라 robot_control_node의
        _on_tf_broadcast_timer가 TF 방송과 함께 수행하고 결과를 self._tcp_pose_cache에
        저장한다(_tcp_tracking_active일 때만) - ToolTrack 콜백(60Hz일 수 있음)마다
        동기 서비스 호출을 하면 여러 요청이 겹쳐 executor 스레드를 점유해 안전상태/
        E-Stop polling이 늦어질 수 있기 때문이다. 이 함수는 캐시만 읽으므로 절대
        블로킹하지 않는다. (2026-07-08: 이전에는 별도 _on_tcp_pose_refresh_timer가
        독립적으로 GetCurrentPosx를 폴링했으나, TF 방송 폴링과 같은 서비스를 이중
        호출해 스레드 고갈을 유발해 하나로 합쳤다.)

        hardware_enabled=false에서는 캐시가 애초에 채워지지 않으므로 항상 None을
        반환한다. 캐시가 없거나 servo_pick.tcp_pose_max_age_s보다 오래됐으면(=조회
        실패가 계속돼 오래된 값을 무한정 재사용하게 되는 경우 포함) None을 반환해
        호출측이 임의의 기본 좌표를 쓰지 않게 한다."""
        if not self.hardware_enabled:
            return None
        cache = self._tcp_pose_cache
        if cache is None:
            return None
        max_age_s = self.get_parameter('servo_pick.tcp_pose_max_age_s').value
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
        return all(math.isfinite(value) for value in (
            position.x, position.y, position.z))

    def _validate_servo_command(self, cmd) -> bool:
        """속도 명령을 실제로 발행하기 직전 마지막 안전 검사. ServoLoop.step()이
        내부적으로 이미 _clip으로 속도를 제한하지만, 발행 경계에서 NaN/Inf와 제한을
        한 번 더 확인해 유효하지 않은 값이 그대로 나가지 않게 한다."""
        values = (cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate)
        if not all(math.isfinite(v) for v in values):
            return False
        tol = self.get_parameter('servo.command_validate_tolerance').value
        v_max = abs(self.get_parameter('servo.v_max').value)
        descend_speed = abs(self.get_parameter('servo.descend_speed').value)
        if abs(cmd.vx) > v_max + tol or abs(cmd.vy) > v_max + tol or abs(cmd.yaw_rate) > v_max + tol:
            return False
        if abs(cmd.vz) > descend_speed + tol:
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
        None)를 반환하는 콜러블 - 현재는 servo_pick(칼만 ServoLoop, TCP pose 필요)만
        이 루프를 쓴다. handover_approach는 movel 기반 단발성 이동으로 별도
        구현될 예정이라 이 루프를 쓰지 않는다(2026-07-07)."""
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
                    if not self._cleanup_stop_motion():
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
        outcome, detail = self._run_rt_tracking(
            goal_handle,
            name='servo_pick',
            message_type=ToolTrack,
            topic='/vision/tool_track',
            callback=self._on_tool_track_during_servo,
            servo=self.servo_loop,
            step=self._servo_pick_step,
            tick=self._servo_pick_tick,
            validate_command=self._validate_servo_command,
            ready_parameter='servo_pick.hardware_ready',
            period_parameter='servo_pick.control_period_s',
            accel_param_prefix='servo_pick')
        if outcome != 'ARRIVED':
            return self._finish_tracking_result(goal_handle, outcome, detail)

        if goal_handle.is_cancel_requested:
            self._cleanup_stop_motion()
            return self._finish_tracking_result(
                goal_handle, 'CANCELED', 'servo_pick canceled before RG2 close')
        if self.safety_state != SafetyState.NORMAL:
            return self._finish_tracking_result(
                goal_handle, 'ABORT',
                f'servo_pick aborted - safety_state={self.safety_state}')

        if not self.rg2_client.close(
                request.grasp_width_mm, request.grasp_force_n,
                goal_handle=goal_handle):
            if self.rg2_client.last_status == RG2Status.CANCELED:
                self._cleanup_stop_motion()
                return self._finish_tracking_result(
                    goal_handle, 'CANCELED', 'servo_pick canceled during RG2 close')
            detail = f'servo_pick RG2 close 실패(status={self.rg2_client.last_status})'
            self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
            return self._finish_tracking_result(goal_handle, 'FAULT', detail)

        width_mm, grip_detected = self.rg2_client.get_state()
        result = self._finish_tracking_result(goal_handle, 'ARRIVED', '')
        result.final_width_mm = width_mm
        result.grip_detected = grip_detected
        return result

    # ---- handover_approach (handover_safe 이후 작업자 손에 접근) ----

    def _on_hand_pose_received(self, msg):
        # 최초 1개만 쓰는 단발성 이동이라 재계산하지 않는다 - 이후 도착하는 메시지는
        # 무시된다(_wait_for_hand_pose가 첫 값을 확인하는 즉시 구독을 정리한다).
        if self._pending_hand_pose is None:
            self._pending_hand_pose = msg

    def _wait_for_hand_pose(self, goal_handle, timeout_s):
        """/vision/hand_pose를 1개 받을 때까지 대기한다 (safety_monitor.wait_for_pull과
        동일한 polling-with-timeout 형태). 결과는 'RECEIVED'/'CANCELED'/'UNSAFE'/
        'TIMEOUT'/'SHUTDOWN' 중 하나 - 실제 메시지는 self._pending_hand_pose에서 읽는다."""
        poll_interval_s = max(self.get_parameter('move.poll_interval_s').value, 0.001)
        deadline = time.monotonic() + timeout_s
        while rclpy.ok():
            if self._pending_hand_pose is not None:
                return 'RECEIVED'
            if goal_handle.is_cancel_requested:
                return 'CANCELED'
            if self.safety_state != SafetyState.NORMAL:
                return 'UNSAFE'
            if time.monotonic() >= deadline:
                return 'TIMEOUT'
            time.sleep(poll_interval_s)
        return 'SHUTDOWN'

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

        if not self.hardware_enabled:
            # dry-run은 다른 _execute_*와 동일하게 센서/하드웨어 의존 없이 흐름만
            # 검증한다 - hand_pose가 아직 vision_node에서 발행되지 않는 환경에서도
            # 실행 경로(취소/성공)를 테스트할 수 있다.
            if self._dry_run_move(goal_handle):
                return self._finish_tracking_result(goal_handle, 'ARRIVED', '')
            if goal_handle.is_cancel_requested:
                return self._finish_tracking_result(
                    goal_handle, 'CANCELED', 'handover_approach canceled (dry_run)')
            return self._finish_tracking_result(
                goal_handle, 'ABORT',
                f'handover_approach aborted (dry_run) - safety_state={self.safety_state}')

        self._pending_hand_pose = None
        subscription = None
        try:
            subscription = self.create_subscription(
                PoseStamped, '/vision/hand_pose', self._on_hand_pose_received, 10,
                callback_group=self.sensor_callback_group)
            wait_outcome = self._wait_for_hand_pose(
                goal_handle, self.get_parameter('handover_approach.timeout_s').value)
            if wait_outcome == 'CANCELED':
                return self._finish_tracking_result(
                    goal_handle, 'CANCELED', 'handover_approach canceled - hand_pose 대기 중')
            if wait_outcome != 'RECEIVED':
                # UNSAFE/TIMEOUT/SHUTDOWN 모두 "계속 진행할 수 없음"이므로 ABORT로
                # 처리한다 - UNSAFE는 이미 safety_monitor가 별도로 FAULT를 선언했을
                # 것이므로 여기서 다시 선언하지 않는다.
                reason = (
                    'handover_approach timeout - hand_pose 미수신'
                    if wait_outcome == 'TIMEOUT' else
                    f'handover_approach aborted - safety_state={self.safety_state}'
                    if wait_outcome == 'UNSAFE' else
                    'handover_approach aborted - rclpy 종료 중')
                return self._finish_tracking_result(goal_handle, 'ABORT', reason)

            hand_pose = self._pending_hand_pose
            expected_frame = self.get_parameter('handover_approach.hand_pose_frame_id').value
            position = hand_pose.pose.position
            if (hand_pose.header.frame_id != expected_frame
                    or not all(
                        math.isfinite(v) for v in (position.x, position.y, position.z))):
                return self._finish_tracking_result(
                    goal_handle, 'ABORT',
                    'handover_approach aborted - hand_pose frame_id/좌표 무효 '
                    f'(frame_id={hand_pose.header.frame_id!r})')

            if goal_handle.is_cancel_requested:
                return self._finish_tracking_result(
                    goal_handle, 'CANCELED', 'handover_approach canceled before TCP 조회')
            if self.safety_state != SafetyState.NORMAL:
                return self._finish_tracking_result(
                    goal_handle, 'ABORT',
                    f'handover_approach aborted - safety_state={self.safety_state}')

            tcp_pos6 = self._doosan.get_current_posx(ref=0)
            if tcp_pos6 is None:
                return self._finish_tracking_result(
                    goal_handle, 'ABORT', 'handover_approach aborted - TCP 위치 조회 실패')

            # hand_pose(m)와 TCP(mm)를 m 단위로 맞춰 방향/거리를 계산한다.
            tcp_m = [value / 1000.0 for value in tcp_pos6[:3]]
            hand_m = [position.x, position.y, position.z]
            delta = [hand_m[i] - tcp_m[i] for i in range(3)]
            distance_m = math.sqrt(sum(d * d for d in delta))

            stop_distance_m = self.get_parameter('handover_approach.stop_distance_m').value
            if distance_m <= stop_distance_m:
                # 이미 정지 거리 이내 - 나눗셈(0-division) 방지 겸, 불필요한 이동을
                # 생략하고 그대로 도착 처리한다.
                return self._finish_tracking_result(
                    goal_handle, 'ARRIVED', 'handover_approach - 이미 stop_distance_m 이내')

            # 목표점 = TCP + (손 방향 벡터) * (1 - stop_distance_m/거리) - 손 앞
            # stop_distance_m 지점에서 멈춘다. 방향각(rx,ry,rz)은 hand_pose의
            # orientation이 아직 의미 없는 identity라 현재 TCP 방향을 그대로 유지한다.
            scale = 1.0 - (stop_distance_m / distance_m)
            target_m = [tcp_m[i] + delta[i] * scale for i in range(3)]
            target_pos6 = [value * 1000.0 for value in target_m] + list(tcp_pos6[3:6])

            vel2 = [
                self.get_parameter('handover_approach.vel_mm_s').value,
                self.get_parameter('move.vel_deg_s').value]
            acc2 = [
                self.get_parameter('handover_approach.acc_mm_s2').value,
                self.get_parameter('move.acc_deg_s2').value]
            success = self._doosan.move_line(
                goal_handle, target_pos6, vel2, acc2, ref=0,
                radius_mm=self.get_parameter('move.blend_radius_mm').value,
                sync_type=self.get_parameter('move.sync_type').value,
                poll_interval_s=self.get_parameter('move.poll_interval_s').value,
                timeout_s=self.get_parameter('move.timeout_s').value)
            # move_named와 동일하게, 이동 함수가 성공을 반환한 직후에도 안전상태를
            # 다시 확인한다 (응답 직후 Fault가 발생하는 경합 방지).
            if success and self.safety_state != SafetyState.NORMAL:
                success = False
            if not success and goal_handle.is_cancel_requested:
                return self._finish_tracking_result(
                    goal_handle, 'CANCELED', 'handover_approach canceled during movel')
            if success:
                return self._finish_tracking_result(goal_handle, 'ARRIVED', '')
            return self._finish_tracking_result(
                goal_handle, 'ABORT', 'handover_approach movel 실패')
        except Exception as exc:
            self.get_logger().error(f'handover_approach 실행 중 예외: {exc}')
            self._cleanup_stop_motion()
            outcome = 'CANCELED' if goal_handle.is_cancel_requested else 'ABORT'
            return self._finish_tracking_result(
                goal_handle, outcome, f'handover_approach exception: {exc}')
        finally:
            self._cleanup_destroy_subscription(subscription)
            self._pending_hand_pose = None

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
                self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                return self._finish_tracking_result(goal_handle, 'FAULT', detail)
            if outcome != 'PULLED':
                return self._finish_tracking_result(
                    goal_handle, 'ABORT', f'handover_hold {outcome.lower()}')

            compliance_ok = self._cleanup_disable_compliance()
            compliance_on = False
            if not compliance_ok:
                detail = 'handover_hold compliance 해제 실패 - RG2를 열지 않습니다.'
                self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                return self._finish_tracking_result(goal_handle, 'FAULT', detail)
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
                self._declare_fault(f'{FaultPrefix.FAULT}{detail}')
                return self._finish_tracking_result(goal_handle, 'FAULT', detail)
            return self._finish_tracking_result(
                goal_handle, 'ARRIVED', 'pull_detected, released')
        except Exception as exc:
            self.get_logger().error(f'handover_hold 실행 중 예외: {exc}')
            self._cleanup_stop_motion()
            outcome = 'CANCELED' if goal_handle.is_cancel_requested else 'ABORT'
            return self._finish_tracking_result(
                goal_handle, outcome, f'handover_hold exception: {exc}')
        finally:
            if compliance_on:
                self._cleanup_disable_compliance()
