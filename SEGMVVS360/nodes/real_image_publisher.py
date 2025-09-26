#!/usr/bin/env python3
import os
import glob
import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import UInt32
from cv_bridge import CvBridge


class RealImagePublisher:
    def __init__(self, image_dir, topic_name="/real_image"):
        self.image_dir = image_dir
        self.topic_name = topic_name
        self.bridge = CvBridge()

        # read controls
        self.skip_stride = max(int(rospy.get_param("~skip_stride", 1)), 1)
        self.max_frames = int(rospy.get_param("~max_frames", -1))

        self.images = self._load_image_paths()

        if not self.images:
            rospy.logerr(f"[Publisher] No images found in directory: {image_dir}")
            rospy.signal_shutdown("No images to publish.")
            return

        self.pub = rospy.Publisher(self.topic_name, Image, queue_size=1, latch=False)
        rospy.Subscriber("/next_frame_trigger", UInt32, self.trigger_callback)

        rospy.set_param("/segmvvs360_node/total_targets", len(self.images))
        rospy.loginfo(
            f"[Publisher] Loaded {len(self.images)} images "
            f"after stride={self.skip_stride}, max_frames={self.max_frames}."
        )
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

    def _publish_image(self, idx):
        if idx >= len(self.images):
            rospy.logwarn(
                f"[Publisher] Requested frame {idx+1}, "
                f"but only {len(self.images)} available."
            )
            return

        image_path = self.images[idx]
        rospy.loginfo(
            f"[Publisher] Publishing frame {idx+1}/{len(self.images)}: "
            f"{os.path.basename(image_path)}"
        )

        cv_img = cv2.imread(image_path)
        if cv_img is None:
            rospy.logwarn(f"[Publisher] Failed to load: {image_path}")
            return

        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        ros_img = self.bridge.cv2_to_imgmsg(cv_img, encoding="rgb8")

        while self.pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.sleep(0.1)

        self.pub.publish(ros_img)

    def trigger_callback(self, msg):
        idx = int(msg.data)
        if idx < 0 or idx >= len(self.images):
            rospy.logwarn(
                f"[Publisher] Requested frame {idx+1}, "
                f"but only {len(self.images)} available."
            )
            return
        self._publish_image(idx)


if __name__ == "__main__":
    rospy.init_node("real_image_publisher")
    image_dir = rospy.get_param("~image_dir", "")
    topic_name = rospy.get_param("~topic_name", "/real_image")
    RealImagePublisher(image_dir, topic_name)
    rospy.spin()
