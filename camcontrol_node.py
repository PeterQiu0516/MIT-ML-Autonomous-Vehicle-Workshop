#!/usr/bin/env python

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import imagefunctions
import rospy
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from pid import PID
from std_msgs.msg import Int8

# Tuning Params
IMG_WIDTH = 128
IMG_HEIGHT = 96

MAX_THROTTLE_GAIN = 0.11
MAX_STEER_GAIN = 1.5707
PID_Kp = 4
PID_Ki = 0
PID_Kd = 0

line_pixel_threshold = 10
wt_dist = 1./2/IMG_WIDTH
wt_ang = 1./2/3.14159

CAMNODE_DT = 0.1
FRAMERATE = 50

# Main Camera Node Class
class CamNode(object):

    def __init__(self):

        # initialize controller
        self.steercontroller = PID(PID_Kp, PID_Ki, PID_Kd, -MAX_STEER_GAIN, MAX_STEER_GAIN)

        # Initial Twist Command
        vel = Twist()
        vel.linear.x = 0
        vel.angular.z = 0
        self.my_twist_command = vel

        # initialize the camera and grab a reference to the raw camera capture
        self.camera = PiCamera()

        # Use capture parameters consistent with camera mode 6, which doesn't crop.
        self.camera.resolution = (640, 480)
        self.camera.framerate = FRAMERATE
        self.camera.sensor_mode = 6

        self.max_linear_vel = MAX_THROTTLE_GAIN
        self.max_angular_vel = MAX_STEER_GAIN

        # search region for line pixels
        self.minx = 0
        self.miny = 0
        self.maxx = 999999
        self.maxy = 999999

        # allow the camera to warmup
        time.sleep(1.0)

        # spin up ros
        rospy.init_node('cam_control_node')
        self.pub = rospy.Publisher('driver_node/cmd_vel', Twist, queue_size=1)

        self.image_pub = rospy.Publisher('camera/image', Image, queue_size=1)
        self.bridge = CvBridge()

        # drive state: stop or go
        self.drive_state = 0
        rospy.Subscriber("driver_node/drivestate", Int8, self.updateDriveState_cb)

        self.loop()

    def updateDriveState_cb(self,state):
        self.drive_state = state.data

    def loop(self):
        dt = CAMNODE_DT
        rate = rospy.Rate(1/dt)

        stopvel = Twist()
        stopvel.linear.x = 0.
        stopvel.angular.z = 0.

        stoptime = -1
        imgcapture = np.zeros((IMG_WIDTH*IMG_HEIGHT*3,), dtype=np.uint8)

        while not rospy.is_shutdown():

            self.camera.capture(imgcapture, 'bgr', use_video_port=True, resize=(IMG_WIDTH, IMG_HEIGHT))
            img = imgcapture.reshape(IMG_HEIGHT, IMG_WIDTH, 3)

            # Get drive commands
            self.twist_from_frame(img, dt)

            # Check drive state - stop or go
           # if (self.drive_state == 1):
               # self.my_twist_command = stopvel

            # shutdown if STOP sign detected
            if (self.drive_state == -1):
                
                self.pub.publish(stopvel)
            elif (self.drive_state == 1):
                print "========= STOP =========="
                # publish drive command
                self.pub.publish(stopvel)
                rospy.signal_shutdown("STOP sign detected")
            elif (self.drive_state == 2):
                print "^^^^^^^^^ WARN ^^^^^^^^^^"
                if (time.time()-stoptime>1.):
                    # pause
                    #self.pub.publish(stopvel)
                    #time.sleep(1.)
                    stoptime = time.time()
                # resume - publish original drive command
                self.pub.publish(self.my_twist_command)
            else:
                # publish drive command
                self.pub.publish(self.my_twist_command)

            rate.sleep()

    def twist_from_frame(self, image, dt):
        # prepare image
        img_warped = imagefunctions.warp(image)
        hsv = cv2.cvtColor(img_warped, cv2.COLOR_BGR2HSV)
        ret, img_bin = cv2.threshold(hsv[:, :, 1], 100, 255, cv2.THRESH_BINARY)

        # pick points for interpolation
        #pts_x, pts_y = imagefunctions.pickpoints(img_bin)
        pts_x, pts_y = imagefunctions.pickpoints2(img_bin, self.minx-10,self.miny-10,self.maxx+10,self.maxy+10)

        # fit polynomial
        if (len(pts_x)>line_pixel_threshold and len(pts_y)>line_pixel_threshold):

            # update search region
            self.minx = min(pts_x)
            self.maxx = max(pts_x)
            self.miny = min(pts_y)
            self.maxy = max(pts_y)

            # fit linr
            z = np.polyfit(pts_y, pts_x, 1)
            p = np.poly1d(z)

            # generate plot coordinates
            ploty = [min(pts_y), max(pts_y)]
            plotx = p(ploty)
            pts = np.stack((plotx, ploty))
            pts = np.transpose(pts)
            pts = pts.reshape((-1, 1, 2))
            ptsplot = pts.astype(int)

            # plot line on image
            lines_img = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2RGB)
            cv2.polylines(lines_img, [ptsplot], False, (0,255,0))
            cv2.line(lines_img, (int(IMG_WIDTH/2), IMG_HEIGHT-1), (int(IMG_WIDTH/2), int(IMG_HEIGHT/2)), (0,0,255), 1)

            out_tile = np.hstack([img_warped, lines_img])

            # publish robot's view
            try:
                imgmsg = self.bridge.cv2_to_imgmsg(out_tile, "bgr8")
                imgmsg.header.stamp = rospy.get_rostime()
                self.image_pub.publish(imgmsg)
            except CvBridgeError as e:
                print(e)

            # cross track error
            dist_to_line = p(IMG_HEIGHT) - (IMG_WIDTH/2) # +ve: line is to the right of car
            slope = z[0] # np.arctan2
            ang_deviation = -slope # +ve: line deviates to right of car

            cte = wt_dist*dist_to_line + wt_ang*ang_deviation

            # Controllers
            throttle = MAX_THROTTLE_GAIN
            steering = self.steercontroller.step(cte, dt)

            # Twist Command
            vel = Twist()
            vel.linear.x = min(self.max_linear_vel, throttle)
            vel.angular.z = steering
            #print 'dist=' + str(dist_to_line) + " ang=" + str(ang_deviation) + " => throttle=" + str(vel.linear.x) + ", steer=" + str(vel.angular.z) + ", state=" + str(self.drive_state)

            # update Twist Command
            self.my_twist_command = vel

        else:
            # publish robot's view
            # plot line on image
            lines_img = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2RGB)
            cv2.line(lines_img, (int(IMG_WIDTH/2), IMG_HEIGHT-1), (int(IMG_WIDTH/2), int(IMG_HEIGHT/2)), (0,0,255), 1)

            out_tile = np.hstack([img_warped, lines_img])

            # publish robot's view
            try:
                imgmsg = self.bridge.cv2_to_imgmsg(out_tile, "bgr8")
                imgmsg.header.stamp = rospy.get_rostime()
                self.image_pub.publish(imgmsg)
            except CvBridgeError as e:
                print(e)

            print 'xxxxxxxxxx LOST LINE xxxxxxxx'

if __name__ == '__main__':
    try:
        CamNode()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start camcontrol node.')
