#!/usr/bin/env python3

import ast
import cv2
import numpy as np
import rospy
import torch
from sensor_msgs.msg import Image
from SEGMVVS360.srv import building_segmentation, building_segmentationResponse
from cv_bridge import CvBridge
from PIL import Image as PILImage
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

# constants
CITYSCAPES_COLORS = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
]

# initialization and load oneformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
model = OneFormerForUniversalSegmentation.from_pretrained(
    "shi-labs/oneformer_cityscapes_swin_large"
).to(device).eval()

def segment_real_image(req):
    try:
        cv_img = CvBridge().imgmsg_to_cv2(req.real_rgb_image, desired_encoding="rgb8")
        pil_img = PILImage.fromarray(cv_img).convert("RGB")
        inputs = processor(pil_img, ["semantic"], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        semantic = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[pil_img.size[::-1]]
        )[0]

        seg_np = semantic.cpu().numpy().astype("uint8")

        target_width = rospy.get_param("~im_width", 320)
        target_height = rospy.get_param("~im_height", 160)

        target_class_ids = rospy.get_param("~target_class_ids", [2])
        if isinstance(target_class_ids, str):
            target_class_ids = ast.literal_eval(target_class_ids)

        dynamic_ids = rospy.get_param("~dynamic_class_ids", [11, 12, 13, 14, 15, 16, 17, 18])
        if isinstance(dynamic_ids, str):
            dynamic_ids = [int(x) for x in ast.literal_eval(dynamic_ids)]

        use_dynamic = rospy.get_param("~use_dynamic_mask", True)

        combined_mask = np.isin(seg_np, target_class_ids).astype(np.uint8) * 255
        resized_mask = cv2.resize(
            combined_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST
        )

        if use_dynamic:
            dynamic_mask = np.isin(seg_np, dynamic_ids).astype(np.uint8) * 255
            dynamic_mask = cv2.resize(
                dynamic_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST
            )
            dyn_iter = rospy.get_param("~dyn_dilate_iter", 1)
            if dyn_iter > 0:
                dynamic_mask = cv2.dilate(
                    dynamic_mask, np.ones((3, 3), np.uint8), iterations=dyn_iter
                )
        else:
            dynamic_mask = np.zeros((target_height, target_width), dtype=np.uint8)

        semantic_rgb = np.array(CITYSCAPES_COLORS, dtype=np.uint8)[
            np.clip(seg_np, 0, len(CITYSCAPES_COLORS) - 1)
        ]
        semantic_resized = cv2.resize(
            semantic_rgb, (target_width, target_height), interpolation=cv2.INTER_NEAREST
        )

        binary_ros_img   = CvBridge().cv2_to_imgmsg(resized_mask, encoding="mono8")
        semantic_ros_img = CvBridge().cv2_to_imgmsg(semantic_resized, encoding="rgb8")
        dynamic_ros_img  = CvBridge().cv2_to_imgmsg(dynamic_mask, encoding="mono8")

        return building_segmentationResponse(
            real_binary_mask_image=binary_ros_img,
            semantic_segmentation_image=semantic_ros_img,
            dynamic_occlusion_image=dynamic_ros_img,
            success=True,
            message="Segmentation completed successfully."
        )

    except Exception as e:
        rospy.logerr(f"Segmentation failed: {str(e)}")
        return building_segmentationResponse(
            real_binary_mask_image=Image(),
            semantic_segmentation_image=Image(),
            dynamic_occlusion_image=Image(),
            success=False,
            message=str(e)
        )

if __name__ == "__main__":
    rospy.init_node("building_segmentation_service")
    service_name = rospy.get_param("~service_name", "building_segmentation")
    rospy.Service(service_name, building_segmentation, segment_real_image)
    rospy.spin()
