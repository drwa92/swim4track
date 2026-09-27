"""Start one disabled controller; the simulator is a separate prerequisite."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    strings = {
        "controller": "learned", "model_path": "", "expected_model_sha256": "",
        "pid_config_dir": "", "trajectory_name": "hold_level", "device": "cpu",
        "odometry_topic": "/bluerov2/odometry_sim", "command_topic": "/bluerov2/setpoint/pwm",
        "expected_command_subscriber": "/stonefish_ros2/stonefish_simulator",
        "expected_world_frame": "", "expected_body_frame": "", "odometry_convention": "ned_frd",
    }
    floats = {
        "acquisition_s": "30.0", "evaluation_s": "60.0", "trajectory_time_scale": "2.0",
        "reference_z_offset_m": "-2.5", "odometry_timeout_s": "0.5",
        "surface_z_m": "0.0", "bottom_z_m": "2.85", "horizontal_bound_m": "3.0",
        "divergence_m": "2.0", "path_speed_ramp_s": "5.0", "command_soft_start_s": "2.0",
    }
    declarations = [DeclareLaunchArgument(k, default_value=v) for k, v in {**strings, **floats}.items()]
    # Explicit string typing also covers paths/names that YAML might parse as numbers.
    parameters = {k: ParameterValue(LaunchConfiguration(k), value_type=str) for k in strings}
    parameters.update({k: ParameterValue(LaunchConfiguration(k), value_type=float) for k in floats})
    parameters["use_sim_time"] = False
    return LaunchDescription(declarations+[
        Node(package="swim4track_ros", executable="controller", name="swim4track_controller",
             output="screen", parameters=[parameters]),
    ])
