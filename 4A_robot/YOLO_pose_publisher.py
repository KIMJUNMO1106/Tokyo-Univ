import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from geometry_msgs.msg import PoseArray, Pose

KEYPOINTS_NAMES = [
    "nose",        # 0
    "eye(L)",      # 1
    "eye(R)",      # 2
    "ear(L)",      # 3
    "ear(R)",      # 4
    "shoulder(L)", # 5
    "shoulder(R)", # 6
    "elbow(L)",    # 7
    "elbow(R)",    # 8
    "wrist(L)",    # 9
    "wrist(R)",    # 10
    "hip(L)",      # 11
    "hip(R)",      # 12
    "knee(L)",     # 13
    "knee(R)",     # 14
    "ankle(L)",    # 15
    "ankle(R)",    # 16
]


class PosePublisherClass(Node):
    def __init__(self):
        super().__init__('simple_face_detection')

        # 1) Initialize YOLO model
        self.model = YOLO("yolov8n-pose.pt")
        self.enableLog = True  # set to False if you don’t want console prints

        # 2) Start Intel RealSense pipeline
        print("# setting color and depth information")
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()

        # Check if any devices
        ctx = rs.context()
        serials = []
        if len(ctx.devices) > 0:
            for dev in ctx.devices:
                print('Found device: ',
                      dev.get_info(rs.camera_info.name), ' ',
                      dev.get_info(rs.camera_info.serial_number))
                serials.append(dev.get_info(rs.camera_info.serial_number))
        else:
            print("No Intel Device connected")

        # Check if we have RGB
        self.found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                self.found_rgb = True
                # 3) Publishers for body + wrist
                self.arrayPublisher_body = self.create_publisher(PoseArray, 'body', 10)
                self.arrayPublisher_Wrist = self.create_publisher(PoseArray, 'wrist', 10)
                break
        if not self.found_rgb:
            print("The demo requires Depth camera with Color sensor.")
            exit(0)

        # Setup realsense alignment + streams
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        # We'll use the device's serial number
        print("# set the serial number")
        self.selialNumber = self.device.get_info(rs.camera_info.serial_number)
        self.config.enable_device(self.selialNumber)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        print("## before starting streaming")
        print(self.config)

        # Start streaming
        print("# start streaming")
        self.pipeline.start(self.config)

        # Get streaming info
        print("## get streaming info")
        self.profile = self.pipeline.get_active_profile()
        self.depth_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.depth))
        print("### set depth intrinsics")
        print(self.depth_profile.get_intrinsics())

        self.depth_intrinsics = self.depth_profile.get_intrinsics()
        self.w, self.h = self.depth_intrinsics.width, self.depth_intrinsics.height

        # Create pointcloud + decimation filter
        print("# set pointcloud")
        self.pc = rs.pointcloud()
        self.decimate = rs.decimation_filter()
        self.decimate.set_option(rs.option.filter_magnitude, 2)
        self.colorizer = rs.colorizer()

        # 4) Timer for main loop
        # Use ~0.1 sec for ~10 FPS
        self.timer = self.create_timer(0.5, self.loop) #2

    def loop(self):
        """
        Called ~every 0.1s. Acquire frames, run YOLO pose detection,
        publish PoseArray messages, and show real-time detection in an OpenCV window.
        """
        # 1) Wait for frames
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return

        depth_frame = self.decimate.process(depth_frame)
        self.depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()
        self.w, self.h = self.depth_intrinsics.width, self.depth_intrinsics.height

        # Convert to numpy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Create a pointcloud
        points = self.pc.calculate(depth_frame)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

        # 2) YOLO inference
        # For safety, resize color_image to w,h if needed
        color_image_s = cv2.resize(color_image, (self.w, self.h))
        results = self.model(color_image_s, show=False, save=False)
        keypoints = results[0].keypoints
        annotated_frame = results[0].plot()

        # 3) We'll gather Pose info for each keypoint
        # We'll store them in arrays we can publish
        id_counter = 1
        positionArray = []
        wristPositionArray = []

        for kp in keypoints:
            if kp.conf is not None:
                if self.enableLog:
                    print("ID:", id_counter)

                # xyn is normalized coords (0..1)
                # shape: keypoints -> [1, num_keypoints, 2]
                kpxy = kp.xyn.tolist()
                kpconf = kp.conf.tolist()

                # 3.1) For each keypoint
                for index in range(len(kpconf[0])):
                    # 2D in color_image_s
                    px = int(kpxy[0][index][0] * self.w)
                    py = int(kpxy[0][index][1] * self.h)

                    # The vertex in the realsense pointcloud
                    # px + (py-1)* self.w might risk out-of-bounds if py=0. So clamp:
                    if py <= 0:
                        py = 1
                    idx_1d = px + (py - 1) * self.w
                    if idx_1d >= len(verts) or idx_1d < 0:
                        continue

                    world_xyz = verts[idx_1d]
                    # Build a Pose with x,y,z = that 3D coordinate
                    poses = Pose()
                    poses.position.x = float(world_xyz[0])
                    poses.position.y = float(world_xyz[1])
                    poses.position.z = float(world_xyz[2])
                    # Store confidence in orientation.x for convenience
                    poses.orientation.x = float(kpconf[0][index])
                    positionArray.append(poses)

                    # Check if it's left or right wrist
                    if index == 9:   # wrist(L)
                        wristPositionArray.append(poses)
                    elif index == 10: # wrist(R)
                        wristPositionArray.append(poses)

                    if self.enableLog:
                        kp_name = KEYPOINTS_NAMES[index]
                        print(f"{id_counter} :world position:{kp_name}: {world_xyz}")

                id_counter += 1

        # 4) Build the PoseArray for body & wrist
        now_time = self.get_clock().now().to_msg()
        header_info = Header()
        header_info.frame_id = "camera_color_optical_frame"
        header_info.stamp = now_time

        msg_body = PoseArray(header=header_info, poses=positionArray)
        msg_wrist = PoseArray(header=header_info, poses=wristPositionArray)

        self.arrayPublisher_body.publish(msg_body)
        self.arrayPublisher_Wrist.publish(msg_wrist)
        # annotated_frame = cv2.rotate(annotated_frame, cv2.ROTATE_180)

        # 5) Show real-time detection window
        cv2.imshow("YOLOv8 Pose", annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)

    publisher = PosePublisherClass()
    rclpy.spin(publisher)

    # Cleanup
    publisher.pipeline.stop()
    publisher.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
