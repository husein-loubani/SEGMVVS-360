#!/usr/bin/env python3
import os
import glob
import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import UInt32
from cv_bridge import CvBridge


class RealImagePublisher:
    def __init__(self, image_dir: str, real_image_topic: str):
        self.image_dir = image_dir
        self.real_image_topic = real_image_topic
        self.bridge = CvBridge()

        # read controls (private params set in launch)
        self.skip_stride = max(int(rospy.get_param("~skip_stride", 1)), 1)
        self.max_frames = int(rospy.get_param("~max_frames", -1))

        self.images = self._load_image_paths()
        if not self.images:
            rospy.logerr(f"[Publisher] No images found in directory: {self.image_dir}")
            rospy.signal_shutdown("No images to publish.")
            return

        self.pub = rospy.Publisher(self.real_image_topic, Image, queue_size=1, latch=False)
        rospy.Subscriber("/next_frame_trigger", UInt32, self.trigger_callback)

        # C++ reads private param: /segmvvs360_node/total_targets
        rospy.set_param("/total_targets", len(self.images))

        rospy.loginfo(
            f"[Publisher] Loaded {len(self.images)} images "
            f"after stride={self.skip_stride}, max_frames={self.max_frames}."
        )
        rospy.loginfo(f"[Publisher] Publishing on: {self.real_image_topic}")
        rospy.loginfo("[Publisher] Waiting for indexed /next_frame_trigger...")

    def _load_image_paths(self):
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        all_paths = sorted(
            f for f in glob.glob(os.path.join(self.image_dir, "*"))
            if f.lower().endswith(valid_ext)
        )
        paths = all_paths[::self.skip_stride]
        if self.max_frames > 0:
            paths = paths[:self.max_frames]
        return paths

    def _publish_image(self, idx: int):
        if idx < 0 or idx >= len(self.images):
            rospy.logwarn(f"[Publisher] Requested frame {idx+1}, but only {len(self.images)} available.")
            return

        image_path = self.images[idx]
        rospy.loginfo(f"[Publisher] Publishing frame {idx+1}/{len(self.images)}: {os.path.basename(image_path)}")

        cv_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if cv_img is None:
            rospy.logwarn(f"[Publisher] Failed to load: {image_path}")
            return

        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        ros_img = self.bridge.cv2_to_imgmsg(cv_img, encoding="rgb8")
        ros_img.header.stamp = rospy.Time.now()

        while self.pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.sleep(0.05)

        self.pub.publish(ros_img)

    def trigger_callback(self, msg: UInt32):
        self._publish_image(int(msg.data))


if __name__ == "__main__":
    rospy.init_node("real_image_publisher")

    image_dir = rospy.get_param("~image_dir", "")
    real_image_topic = rospy.get_param("~real_image_topic", "/real_image")

    RealImagePublisher(image_dir, real_image_topic)
    rospy.spin()
