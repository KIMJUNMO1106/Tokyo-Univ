#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node

# For the PoseArray from /wrist
from geometry_msgs.msg import PoseArray, PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# MoveIt IK service
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import MoveItErrorCodes

# TF2
import tf2_ros
from tf2_geometry_msgs import do_transform_pose_stamped


class WristToRobot(Node):
    def __init__(self):
        super().__init__('wrist_to_robot')

        # 1) Setup TF2
        self.get_logger().info("[Init] Setting up TF2 (transform from camera to link_base)...")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        # 2) Initialize the /compute_ik service client (MoveIt)
        self.get_logger().info("[Init] Waiting for /compute_ik service...")
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        if not self.ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("...still waiting for /compute_ik service")
            return

        # 3) Publisher to /lite6_traj_controller/joint_trajectory
        self.get_logger().info("[Init] Creating publisher to /lite6_traj_controller/joint_trajectory")
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/lite6_traj_controller/joint_trajectory',
            10
        )

        # 4) Subscribe to /wrist
        self.get_logger().info("[Init] Subscribing to /wrist (PoseArray) for new poses...")
        self.sub_wrist = self.create_subscription(
            PoseArray,
            '/wrist',
            self.wrist_callback,
            10
        )

        self.get_logger().info("WristToRobot node started. Now listening for /wrist messages...")

    def wrist_callback(self, msg: PoseArray):
        """
        1) Takes the first pose in the PoseArray
        2) Transforms it from the camera frame to link_base
        3) Calls IK
        4) If there's a valid IK solution, publishes a trajectory
        5) If no solution, skip it
        """
        self.get_logger().info(f"Received PoseArray with {len(msg.poses)} poses.")
        if not msg.poses:
            self.get_logger().warn("PoseArray is empty. Skipping.")
            return

        # 1) Convert the first pose to a PoseStamped
        wrist_pose_cam = PoseStamped()
        wrist_pose_cam.header = msg.header  # e.g. "camera_color_optical_frame"
        wrist_pose_cam.pose = msg.poses[0]

        # 2) Transform to link_base
        transformed_pose = self.transform_pose(wrist_pose_cam, "link_base")
        if transformed_pose is None:
            self.get_logger().warn("Skipping because transform_pose returned None.")
            return

        # 3) Solve IK
        joint_positions = self.call_ik(transformed_pose)
        if joint_positions is None:
            self.get_logger().warn("No IK solution for this pose. Skipping.")
            return

        # 4) If solution found, publish
        self.send_trajectory(joint_positions)

    def transform_pose(self, pose_stamped, target_frame):
        """
        Attempt to transform the pose from pose_stamped.header.frame_id to 'target_frame'.
        Return the transformed PoseStamped or None on failure.
        """
        from rclpy.duration import Duration

        try:
            can_tf = self.tf_buffer.can_transform(
                target_frame,
                pose_stamped.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0)
            )
            if not can_tf:
                self.get_logger().error(f"Cannot TF from {pose_stamped.header.frame_id} to {target_frame}")
                return None

            transform = self.tf_buffer.lookup_transform(
                target_frame,
                pose_stamped.header.frame_id,
                rclpy.time.Time()
            )
            return do_transform_pose_stamped(pose_stamped, transform)

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Transform error: {e}")
            return None

    def call_ik(self, pose_in_base: PoseStamped):
        """
        Call MoveIt's /compute_ik for the 'lite6' group, link_eef as end-effector.
        If successful, return [joint1..joint6] in radians. Otherwise None.
        """
        from moveit_msgs.msg import PositionIKRequest
        from builtin_interfaces.msg import Duration
        pose_in_base.pose.position.x = 0.2
        pose_in_base.pose.position.y = 0.0
        pose_in_base.pose.position.z = 0.2
        
        pose_in_base.pose.orientation.x = 1.0
        pose_in_base.pose.orientation.y = 0.0
        pose_in_base.pose.orientation.z = 0.0
        pose_in_base.pose.orientation.w = 0.0
        # print(pose_in_base)
        # Prepare the IK request
        ik_req = GetPositionIK.Request()
        ik_req.ik_request = PositionIKRequest()
        ik_req.ik_request.group_name = "lite6"
        ik_req.ik_request.avoid_collisions = False
        ik_req.ik_request.ik_link_name = "link_eef"
        ik_req.ik_request.pose_stamped = pose_in_base
        ik_req.ik_request.timeout = Duration(sec=2)

        # Provide a seed state of all zeros
        seed = JointState()
        seed.name = ['joint1','joint2','joint3','joint4','joint5','joint6']
        seed.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ik_req.ik_request.robot_state.joint_state = seed

        self.get_logger().info(f"Requesting IK for pose (x={pose_in_base.pose.position.x:.3f},"
                               f" y={pose_in_base.pose.position.y:.3f},"
                               f" z={pose_in_base.pose.position.z:.3f})")
        


        # Call service
        self.get_logger().info("Calling /compute_ik...")
        # print(ik_req)
        future = self.ik_client.call_async(ik_req)
        # print(future)
        self.get_logger().info("Check0")
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info("Check1")
        resp = future.result()

        if not resp:
            self.get_logger().error("IK call returned no response!")
            return None
        
        

        # Check if success
        if resp.error_code.val == MoveItErrorCodes.SUCCESS:
            # parse angles from resp.solution.joint_state
            angle_map = dict(zip(resp.solution.joint_state.name,
                                 resp.solution.joint_state.position))
            # build a list [joint1..joint6]
            angles = [
                angle_map['joint1'],
                angle_map['joint2'],
                angle_map['joint3'],
                angle_map['joint4'],
                angle_map['joint5'],
                angle_map['joint6'],
            ]
            # log them
            self.get_logger().info(f"IK solution (rad): {angles}")
            # deg = [math.degrees(a) for a in angles]
            # self.get_logger().info(f"IK solution rad={angles}, deg={deg}")
            return angles
        else:
            self.get_logger().error(f"IK failed, error code={resp.error_code.val}")
            return None

    def send_trajectory(self, joint_positions):
        """
        Publish a single-point trajectory to /lite6_traj_controller/joint_trajectory.
        The arm then moves in ~ 'time_from_start.sec' seconds.
        """
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6']

        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start.sec = 5  # 5 seconds
        traj.points.append(point)

        self.traj_pub.publish(traj)
        self.get_logger().info("Published single-point trajectory to move the arm.")


def main(args=None):
    rclpy.init(args=args)
    node = WristToRobot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


