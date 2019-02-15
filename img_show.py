#!/usr/bin/env python

import cv2, rospy, matplotlib.pyplot as plt, numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()


rospy.init_node('img_show', anonymous=True)

def callback(msg):
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    cv2.imshow('cv', cv_image)
    cv2.waitKey(1)

sub = rospy.Subscriber('/camera/image', Image, callback)

rospy.spin()
