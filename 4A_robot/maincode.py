#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import MoveItErrorCodes

import tf2_ros
from tf2_geometry_msgs import do_transform_pose_stamped


class WristToRobotAsync(Node):
    def __init__(self):
        super().__init__('wrist_to_robot_async')

        ########################
        # TF2 Setup
        ########################
        self.get_logger().info("[Init] TF2 for camera->link_base transform...")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=True
        )

        ########################
        # IK Service Client
        ########################
        self.get_logger().info("[Init] Connecting to /compute_ik...")
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        if not self.ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Can't connect to /compute_ik after 5s.")
            return

        ########################
        # Trajectory Publisher
        ########################
        self.get_logger().info("[Init] Publisher for /lite6_traj_controller/joint_trajectory")
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/lite6_traj_controller/joint_trajectory',
            10
        )

        ########################
        # Pose Subscriber
        ########################
        self.get_logger().info("[Init] Subscribing to /wrist for PoseArray")
        self.sub_wrist = self.create_subscription(
            PoseArray,
            '/wrist',
            self.wrist_callback,
            10
        )

        ########################
        # Internal storage
        ########################
        self.latest_pose_in_base = None
        self.last_pose_time = None

        ########################
        # Timer to check new pose & call IK
        ########################
        self.timer = self.create_timer(1.0, self.timer_cb)

        self.get_logger().info("Node started. Listening to /wrist & performing IK outside the callback.")

    def wrist_callback(self, msg: PoseArray):
        if not msg.poses:
            return

        # We'll only transform the first pose
        pose_cam = PoseStamped()
        pose_cam.header = msg.header
        pose_cam.pose = msg.poses[0]

        # Try to transform to link_base
        pose_base = self.transform_pose(pose_cam, 'link_base')
        # Right after transform_pose(...) returns pose_base and before you store/use it, do:

        if pose_base is None:
            return

        # Store it
        self.latest_pose_in_base = pose_base
        self.last_pose_time = self.get_clock().now()
        self.get_logger().info("Got new wrist pose in link_base. Will handle in timer callback...")

    def timer_cb(self):
        """
        Called every 0.5s. If we have a new pose, call IK async.
        """
        if self.latest_pose_in_base is not None:
            # We'll handle the pose once, then clear it so we don't spam.
            pose = self.latest_pose_in_base
            self.latest_pose_in_base = None

            # Build the IK request & do an async call
            req = self.build_ik_request(pose)
            self.get_logger().info("Sending IK request asynchronously...")
            future = self.ik_client.call_async(req)
            future.add_done_callback(self.ik_response_cb)
        # else no new pose, do nothing

    def ik_response_cb(self, future):
        """
        Called once the IK solver finishes. We are in a non-callback thread.
        """
        rclpy.spin_until_future_complete(self, future)
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(f"IK service failed: {e}")
            return

        if not resp:
            self.get_logger().error("No response from IK solver!")
            return

        if resp.error_code.val == MoveItErrorCodes.SUCCESS:
            # parse angles
            angle_map = dict(zip(resp.solution.joint_state.name,
                                 resp.solution.joint_state.position))
            # build [joint1..joint6]
            angles = [
                angle_map['joint1'],
                angle_map['joint2'],
                angle_map['joint3'],
                angle_map['joint4'],
                angle_map['joint5'],
                angle_map['joint6'],
            ]
            self.get_logger().info(f"IK success. angles rad={angles}")
            self.send_trajectory(angles)
        else:
            self.get_logger().warn(f"IK solver failed. error_code={resp.error_code.val}")

    def build_ik_request(self, target_pose: PoseStamped):
        """
        Return a GetPositionIK request for the given PoseStamped in link_base frame.
        """
        from moveit_msgs.msg import PositionIKRequest
        from builtin_interfaces.msg import Duration

        target_pose.pose.orientation.x = 1.0
        target_pose.pose.orientation.y = 0.0
        target_pose.pose.orientation.z = 0.0
        target_pose.pose.orientation.w = 0.0

        # target_pose.pose.position.x = 0.2
        # target_pose.pose.position.y = 0.0
        # target_pose.pose.position.z = 0.2


        ik_req = GetPositionIK.Request()
        ik_req.ik_request = PositionIKRequest()
        ik_req.ik_request.group_name = "lite6"
        ik_req.ik_request.avoid_collisions = False
        ik_req.ik_request.ik_link_name = "link_eef"
        ik_req.ik_request.pose_stamped = target_pose
        ik_req.ik_request.timeout = Duration(sec=2)

        seed = JointState()
        seed.name = ['joint1','joint2','joint3','joint4','joint5','joint6']
        seed.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ik_req.ik_request.robot_state.joint_state = seed
        self.get_logger().info(f"Requesting IK for pose (x={target_pose.pose.position.x:.3f},"
                               f" y={target_pose.pose.position.y:.3f},"
                               f" z={target_pose.pose.position.z:.3f})")

        return ik_req

    def transform_pose(self, pose_stamped, target_frame):
        from rclpy.duration import Duration
        try:
            can_tf = self.tf_buffer.can_transform(
                target_frame,
                pose_stamped.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0)
            )
            if not can_tf:
                self.get_logger().error(
                    f"Cannot transform from {pose_stamped.header.frame_id} to {target_frame}"
                )
                return None

            tfm = self.tf_buffer.lookup_transform(
                target_frame,
                pose_stamped.header.frame_id,
                rclpy.time.Time()
            )
            return do_transform_pose_stamped(pose_stamped, tfm)

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Transform error: {e}")
            return None

    def send_trajectory(self, joint_positions):
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6']

        pt = JointTrajectoryPoint()
        pt.positions = joint_positions
        pt.time_from_start.sec = 2 #5
        traj.points.append(pt)

        self.traj_pub.publish(traj)
        self.get_logger().info("Single-point trajectory published. The robot moves in ~5s.")


def main(args=None):
    rclpy.init(args=args)
    node = WristToRobotAsync()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

