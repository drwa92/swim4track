"""Enable one ready demo and preserve its status until completion or abort."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import SetBool


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--readiness-timeout", type=float, default=90.0)
    ns = parser.parse_args(args)
    for name in ("timeout", "readiness_timeout"):
        value = getattr(ns, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    ns.output.mkdir(parents=True, exist_ok=True)
    trace = (ns.output/"status.jsonl").open("x")
    rclpy.init(args=[])
    node = Node("swim4track_demo_watch")
    last = {}
    identity = {}
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)

    def status_callback(msg):
        nonlocal last
        last = json.loads(msg.data)
        trace.write(json.dumps(dict(received_monotonic_s=time.monotonic(), **last))+"\n")
        trace.flush()

    def identity_callback(msg):
        nonlocal identity
        identity = json.loads(msg.data)

    sub = node.create_subscription(String, "/swim4track/status", status_callback, qos)
    ident_sub = node.create_subscription(String, "/swim4track/identity", identity_callback, qos)
    client = node.create_client(SetBool, "/swim4track/set_enabled")
    activated = False
    code = 1
    result = {"outcome": "infrastructure_failure"}
    try:
        ready_deadline = time.monotonic()+ns.readiness_timeout
        while time.monotonic() < ready_deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            if client.service_is_ready() and last.get("ready") and identity:
                break
        else:
            raise RuntimeError("readiness timeout: "+last.get("readiness_reason", "no status received"))
        (ns.output/"identity.json").write_text(json.dumps(identity, indent=2)+"\n")
        prior_run_id = last.get("run_id")
        future = client.call_async(SetBool.Request(data=True))
        rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
        response = future.result() if future.done() else None
        if response is None or not response.success:
            raise RuntimeError("enable request failed: "+(response.message if response else "timeout"))
        activated = True
        run_id = None
        deadline = time.monotonic()+ns.timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            # Status is transient-local: a previous run's terminal status can
            # arrive around the enable response. Only accept this new run.
            observed_id = last.get("run_id")
            if not observed_id or observed_id == prior_run_id:
                continue
            if run_id is None:
                run_id = observed_id
            if observed_id != run_id:
                raise RuntimeError("another start request replaced the watched run")
            if last.get("phase") == "completed":
                result, code = dict(outcome="completed", final_status=last), 0
                break
            if last.get("phase") in {"aborted", "disabled"}:
                result = dict(outcome="aborted", final_status=last)
                break
        else:
            result = dict(outcome="timeout", final_status=last)
    except KeyboardInterrupt:
        result = dict(outcome="interrupted", final_status=last)
        code = 130
    except Exception as exc:
        result = dict(outcome="infrastructure_failure", error=str(exc), final_status=last)
    finally:
        # Store the outcome before network teardown: a repeated Ctrl-C or a
        # stopped ROS context must not recreate the historical missing-file case.
        outcome_path = ns.output/"outcome.json"
        outcome_path.write_text(json.dumps(result, indent=2)+"\n")
        try:
            if activated and rclpy.ok() and client.service_is_ready():
                future = client.call_async(SetBool.Request(data=False))
                rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)
        except (Exception, KeyboardInterrupt) as exc:
            result["shutdown_note"] = f"disable request interrupted: {type(exc).__name__}: {exc}"
        finally:
            try:
                outcome_path.write_text(json.dumps(result, indent=2)+"\n")
            finally:
                trace.close()
                try:
                    node.destroy_node()
                except (Exception, KeyboardInterrupt):
                    pass
                try:
                    if rclpy.ok():
                        rclpy.shutdown()
                except (Exception, KeyboardInterrupt):
                    pass
    print(json.dumps(result, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
