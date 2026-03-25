from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import GroupAction, IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    realsense_launch_dir = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch'
    )

    return LaunchDescription([
        GroupAction([
            PushRosNamespace('left_camera'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(realsense_launch_dir, 'rs_launch.py')
                ),
                launch_arguments={
                    'serial_no': '_335222070270',
        
                    # Left camera: human pose estimation
                    'enable_color': 'true',
                    'enable_depth': 'true',
                    'enable_gyro': 'false',
                    'enable_accel': 'false',
                    'align_depth.enable': 'true',
        
                    'depth_module.depth_profile': '848x480x30',
                    'rgb_camera.color_format': 'rgb8',
                    'rgb_camera.color_profile': '848x480x30',
                    'rgb_camera.enable_auto_exposure': 'false',
                    'rgb_camera.exposure': '75',
                    'rgb_camera.gain': '80',
        
                    'frames_queue_size': '8',
        
                    # Disable unnecessary modules and outputs
                    'enable_infra1': 'false',
                    'enable_infra2': 'false',
                    'decimation_filter.enable': 'false',
                    'enable_sync': 'false',
                    'enable_pointcloud': 'false',
                    'enable_color_pointcloud': 'false',
                    'enable_depth_to_infra1': 'false',
                    'enable_depth_to_infra2': 'false',
                    'enable_depth_to_rgb': 'false',
                    'publish_tf': 'false',
                    'publish_tf_static': 'false',
                    'enable_compressed': 'false',
                    'enable_color_compression': 'false',
                    'enable_depth_compression': 'false'
                }.items(),
            ),
        ]),

        GroupAction([
            PushRosNamespace('right_camera'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(realsense_launch_dir, 'rs_launch.py')
                ),
                launch_arguments={
                    'serial_no': '_018322070277',

                    # Right camera: ArUco / ChArUco tracking
                    'enable_color': 'true',
                    'enable_depth': 'true',
                    'enable_gyro': 'false',
                    'enable_accel': 'false',
                    'align_depth.enable': 'true',

                    # High-res RGB, lower-res depth
                    'depth_module.depth_profile': '1280x720x30',
                    'rgb_camera.color_format': 'rgb8',
                    'rgb_camera.color_profile': '1280x720x30',
                    'rgb_camera.enable_auto_exposure': 'false',
                    'rgb_camera.exposure': '75',
                    'rgb_camera.gain': '80',

                    'enable_sync': 'true',

                    'frames_queue_size': '8',

                    # Disable unnecessary modules and outputs
                    'enable_infra1': 'false',
                    'enable_infra2': 'false',
                    'decimation_filter.enable': 'false',
                    'enable_pointcloud': 'false',
                    'enable_color_pointcloud': 'false',
                    'enable_depth_to_infra1': 'false',
                    'enable_depth_to_infra2': 'false',
                    'enable_depth_to_rgb': 'false',
                    'publish_tf': 'false',
                    'publish_tf_static': 'false',
                    'enable_compressed': 'false',
                    'enable_color_compression': 'false',
                    'enable_depth_compression': 'false'
                }.items(),
            )
        ]),

        Node(
            package='pt_collect',
            executable='aruco_detector',
            name='aruco_detector',
            output='screen',
        ),
    ])