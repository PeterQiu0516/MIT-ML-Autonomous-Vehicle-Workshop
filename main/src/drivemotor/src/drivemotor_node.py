#!/usr/bin/python

DRIVENODE_DT = 0.05

from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor

import rospy
import time
import atexit
import math

from geometry_msgs.msg import Twist

# create a default object, no changes to I2C address or frequency
mh = Adafruit_MotorHAT(addr=0x60)

# recommended for auto-disabling motors on shutdown!
def turnOffMotors():
    mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)

atexit.register(turnOffMotors)

#################################
motor1 = mh.getMotor(1) # front left
motor2 = mh.getMotor(2) # front right
motor3 = mh.getMotor(3) # rear left
motor4 = mh.getMotor(4) # rear right

class MotorDriver():

  def __init__(self):
    #set initial speed, from 0 (off) to 255 (max speed)
    motor1.setSpeed(150)
    motor2.setSpeed(150)
    motor3.setSpeed(150)
    motor4.setSpeed(150)

    motor1.run(Adafruit_MotorHAT.FORWARD);
    motor2.run(Adafruit_MotorHAT.FORWARD);
    motor3.run(Adafruit_MotorHAT.FORWARD);
    motor4.run(Adafruit_MotorHAT.FORWARD);

    # turn on motor
    motor1.run(Adafruit_MotorHAT.RELEASE);
    motor2.run(Adafruit_MotorHAT.RELEASE);
    motor3.run(Adafruit_MotorHAT.RELEASE);
    motor4.run(Adafruit_MotorHAT.RELEASE);

    # initialize motor settings
    self.left_wheel = 0
    self.right_wheel = 0

    rospy.init_node('motordriver', anonymous=True)
    rospy.Subscriber("driver_node/cmd_vel", Twist, self.callback)
    self.loop()
    #rospy.spin()

  def callback(self,twist):
    max_vel = 0.2 # set in joystick control node
    max_angle = 1.5707
    wheeldist = 0.1
    gain  = 255 / (max_vel + (max_angle * wheeldist)) # 700
    self.left_wheel = math.trunc(gain * (twist.linear.x + twist.angular.z * wheeldist))
    self.right_wheel = math.trunc(gain * (twist.linear.x - twist.angular.z * wheeldist))

    #rospy.loginfo("twist.x: " + str(twist.linear.x) + " twist.az: " + str(twist.angular.z) + " left: " + str(self.left_wheel) + " right: " + str(self.right_wheel) )

  def loop(self):

    dt = DRIVENODE_DT
    rate = rospy.Rate(1/dt)

    while not rospy.is_shutdown():

      #left wheels
      if self.left_wheel>0:
        motor1.run(Adafruit_MotorHAT.FORWARD)
        motor3.run(Adafruit_MotorHAT.FORWARD)
        motor1.setSpeed(self.left_wheel)
        motor3.setSpeed(self.left_wheel)
      else:
        motor1.run(Adafruit_MotorHAT.BACKWARD)
        motor3.run(Adafruit_MotorHAT.BACKWARD)
        motor1.setSpeed(-self.left_wheel)
        motor3.setSpeed(-self.left_wheel)

      #right wheels
      if self.right_wheel>0:
        motor2.run(Adafruit_MotorHAT.FORWARD)
        motor4.run(Adafruit_MotorHAT.FORWARD)
        motor2.setSpeed(self.right_wheel)
        motor4.setSpeed(self.right_wheel)
      else:
        motor2.run(Adafruit_MotorHAT.BACKWARD)
        motor4.run(Adafruit_MotorHAT.BACKWARD)
        motor2.setSpeed(-self.right_wheel)
        motor4.setSpeed(-self.right_wheel)

      rate.sleep()

if __name__ == '__main__':
  MotorDriver()
