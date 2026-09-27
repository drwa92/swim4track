"""Frozen learned/PID inference for the audited Stonefish NED/FRD interface."""
from __future__ import annotations

import json
import time
from dataclasses import fields

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import SetBool

from swim4track.runtime import RuntimeConfig, TrackingRuntime


class ControllerNode(Node):
    def __init__(self):
        super().__init__("swim4track_controller")
        defaults = dict(
            controller="learned", model_path="", expected_model_sha256="", pid_config_dir="",
            device="cpu", odometry_topic="/bluerov2/odometry_sim",
            command_topic="/bluerov2/setpoint/pwm", odometry_convention="ned_frd",
            expected_command_subscriber="/stonefish_ros2/stonefish_simulator",
            expected_world_frame="", expected_body_frame="",
        )
        defaults.update({f.name: getattr(RuntimeConfig(), f.name)
                         for f in fields(RuntimeConfig) if f.name != "controller_kind"})
        for name, value in defaults.items():
            self.declare_parameter(name, value, ParameterDescriptor(read_only=True))
        get = lambda name: self.get_parameter(name).value
        if get("odometry_convention") != "ned_frd":
            raise ValueError("only explicitly audited NED pose / FRD body twist is supported")
        if self.get_parameter("use_sim_time").value:
            raise ValueError("this real-time Stonefish demo requires use_sim_time=false")
        kind = str(get("controller"))
        if kind == "learned":
            import torch
            from swim4track.policy import FrozenPolicy
            torch.set_num_threads(1)
            controller = FrozenPolicy(
                model_path=get("model_path") or None,
                expected_sha256=get("expected_model_sha256") or None,
                device=str(get("device")),
            )
            metadata = controller.metadata
        elif kind == "pid":
            from swim4track.pid import load_frozen_pid
            controller, metadata = load_frozen_pid(get("pid_config_dir") or None)
        else:
            raise ValueError("controller must be learned or pid")
        cfg = RuntimeConfig(controller_kind=kind, **{
            f.name: get(f.name) for f in fields(RuntimeConfig) if f.name != "controller_kind"
        })
        self.runtime = TrackingRuntime(controller, cfg)
        self.command_topic = str(get("command_topic"))
        self.expected_command_subscriber = str(get("expected_command_subscriber"))
        if not self.expected_command_subscriber.startswith("/"):
            raise ValueError("expected_command_subscriber must be a fully qualified node name")
        self.expected_world_frame = str(get("expected_world_frame"))
        self.expected_body_frame = str(get("expected_body_frame"))
        self.observed_frames = None
        self.last_source_stamp_ns = None
        self.command_pub = self.create_publisher(Float64MultiArray, self.command_topic, 10)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, "/swim4track/status", latched)
        self.identity_pub = self.create_publisher(String, "/swim4track/identity", latched)
        self.publishers = {
            name: self.create_publisher(Float64MultiArray, "/swim4track/"+name, 10)
            for name in ("action", "desired_pwm", "reference_state", "controller_timing")
        }
        self.subscription = self.create_subscription(
            Odometry, str(get("odometry_topic")), self.on_odometry, qos_profile_sensor_data)
        self.service = self.create_service(SetBool, "/swim4track/set_enabled", self.on_enable)
        # Both execution and stale-data checks continue even if the ROS clock pauses.
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(cfg.control_dt_s, self.on_tick, clock=self.steady_clock)
        identity = dict(schema="swim4track_demo_identity_v1", controller=kind,
                        controller_metadata=metadata, runtime=cfg.__dict__,
                        odometry_convention="NED pose, FRD body twist", simulator_only=True,
                        source_scheduler="monotonic demo runtime; not historical campaign replay",
                        interface=dict(command_signs=[-1]*8, mapping="force_equivalent",
                                       training_command_limit=0.60, time_constant_s=0.25,
                                       rate_limit_per_s=2.0))
        self.identity_pub.publish(String(data=json.dumps(identity, sort_keys=True)))
        self.publish_status()
        self.get_logger().info("Loaded frozen controller; disabled. Start with /swim4track/set_enabled.")

    def graph_ready(self):
        if self.count_publishers(self.command_topic) != 1:
            return False, "command topic must have exactly one publisher (this node)"
        subscribers = {
            "/"+"/".join(p for p in (info.node_namespace.strip("/"), info.node_name) if p)
            for info in self.get_subscriptions_info_by_topic(self.command_topic)
        }
        if self.expected_command_subscriber not in subscribers:
            return False, "expected Stonefish command subscriber is absent"
        if not self.runtime.odometry_fresh(time.monotonic()):
            return False, "missing or stale valid odometry"
        return True, "ready"

    def publish_status(self, status=None):
        ready, reason = self.graph_ready()
        data = dict(status or self.runtime.status(), ready=ready, readiness_reason=reason,
                    observed_frames=self.observed_frames)
        self.status_pub.publish(String(data=json.dumps(data, sort_keys=True, allow_nan=False)))

    def publish_result(self, result):
        if result.command is not None:
            self.command_pub.publish(Float64MultiArray(data=result.command.tolist()))
        for key, value in (("action", result.action), ("desired_pwm", result.desired_command)):
            if value is not None:
                self.publishers[key].publish(Float64MultiArray(data=value.tolist()))
        if result.reference is not None:
            r = result.reference
            self.publishers["reference_state"].publish(Float64MultiArray(data=np.r_[
                r.position_ned, r.rpy, r.linear_velocity_ned, r.angular_velocity_ned].tolist()))
        if result.compute_ms is not None:
            self.publishers["controller_timing"].publish(Float64MultiArray(data=[
                self.runtime.elapsed_s, result.compute_ms,
                float(result.compute_ms >= self.runtime.config.control_dt_s*1000.0)]))
        self.publish_status(result.status)

    def on_odometry(self, msg):
        frames = (msg.header.frame_id, msg.child_frame_id)
        if ((self.expected_world_frame and frames[0] != self.expected_world_frame)
                or (self.expected_body_frame and frames[1] != self.expected_body_frame)):
            if self.runtime.active:
                self.publish_result(self.runtime.abort("odometry frame mismatch"))
            return
        if self.observed_frames is not None and list(frames) != self.observed_frames and self.runtime.active:
            self.publish_result(self.runtime.abort("odometry frame identifiers changed during operation"))
            return
        stamp_ns = msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
        # Do not let repeatedly delivered stamped samples refresh the watchdog.
        if (stamp_ns > 0 and self.last_source_stamp_ns is not None
                and stamp_ns <= self.last_source_stamp_ns and self.runtime.active):
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        quaternion = np.array([q.x, q.y, q.z, q.w], dtype=float)
        if not np.all(np.isfinite(quaternion)) or np.linalg.norm(quaternion) < 1e-9:
            if self.runtime.active:
                self.publish_result(self.runtime.abort("invalid odometry quaternion"))
            return
        xyz = Rotation.from_quat(quaternion).as_euler("xyz")
        v, w = msg.twist.twist.linear, msg.twist.twist.angular
        state = np.array([p.x, p.y, p.z, *xyz, v.x, v.y, v.z, w.x, w.y, w.z])
        if self.runtime.update_odometry(state, time.monotonic()):
            self.observed_frames = list(frames)
            self.last_source_stamp_ns = stamp_ns if stamp_ns > 0 else None
        elif self.runtime.active:
            self.publish_result(self.runtime.abort("nonfinite odometry"))

    def on_enable(self, request, response):
        if not request.data:
            self.publish_result(self.runtime.stop())
            response.success, response.message = True, "disabled; active output stopped"
            return response
        ready, reason = self.graph_ready()
        if not ready:
            response.success, response.message = False, "start refused: "+reason
            return response
        response.success, response.message = self.runtime.start(time.monotonic())
        self.publish_status()
        return response

    def on_tick(self):
        if self.runtime.active:
            ready, reason = self.graph_ready()
            if not ready:
                self.publish_result(self.runtime.abort(reason))
                self.get_logger().error("Controller aborted: "+reason)
                return
        self.publish_result(self.runtime.step(time.monotonic()))

    def stop_safely(self):
        result = self.runtime.stop("node shutdown")
        if self.context.ok() and result.command is not None:
            try:
                self.publish_result(result)
            except Exception as exc:
                self.get_logger().warning(f"Could not publish shutdown zero: {exc}")


def main(args=None):
    # Keep the context alive during Ctrl-C so a final zero can be submitted.
    # SIGKILL/process crashes still require a downstream plant/driver watchdog.
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = ControllerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.stop_safely()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
