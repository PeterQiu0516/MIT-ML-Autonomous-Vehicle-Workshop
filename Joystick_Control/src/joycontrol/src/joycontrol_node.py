#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy


class JoyNode(object):
    def __init__(self):
        rospy.init_node('robocar_joy_teleop')

        rospy.Subscriber("joy", Joy, self.callback)
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        self.throttle = 0.0
        self.steer = 0.0
        self.twist = Twist()

        self.max_linear_vel = rospy.get_param('~max_linear_vel', 0.2)
        self.max_angular_vel = rospy.get_param('~max_angular_vel', 1.5707)

        self.loop()

    def callback(self, data):
        self.throttle = 2*(-data.axes[5]+data.axes[2])
        self.steer = -data.axes[0]
        # right trigger (axes 5) = accelerate
        # left trigger (axes 2) = reverse / brake
        # left joystick (axes 0) = steer

        #######################################################
        # TO-DO:
        # Assign values to self.throttle and self.steer
        # notes:
        #    - access values by the axes, e.g. data.axes[5]
        #    - keep self.throttle values range from -1 to 1
        #    - keep self.steer value range from -1 to 1
        #######################################################

    def loop(self):

        dt = 0.02
        rate = rospy.Rate(1/dt)

        while not rospy.is_shutdown():
            # print 'throttle: ' + str(self.throttle) + ', steer: ' + str(self.steer)
            self.twist.linear.x = self.max_linear_vel * self.throttle
            self.twist.angular.z = self.max_angular_vel * self.steer
            self.pub.publish(self.twist)
            rate.sleep()


if __name__ == '__main__':
    JoyNode()
