#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry

import math
import collections 

from collections import deque

class process(object):
    def __init__(self):


        self.vars = None
        self.varsaw = None
        self.prediction = None

        self.d = deque('0000000000')

        self.subscriber_odom = rospy.Subscriber("odom", Odometry, self.odom_callback)

        # self.vars = 0
        # self.varsaw = 0
        # self.prediction = 0       

        rospy.Subscriber("/lin_cord", Point, self.lidar_callback)
        rospy.Subscriber("/predic", Point, self.predic_callback, queue_size=1)
        rospy.Subscriber("/ang_cord",Point,self.ang_callback, queue_size=1)

        self.publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=5)

        self.Init = True
        self.Init_ang = None

        self.Init_pos = Point()
        self.Init_pos.x = 0
        self.Init_pos.y = 0
        self.Init_pos.z = 0

        self.globalAng = 0
        self.globalPos = Point()
        self.globalPos.x = 0
        self.globalPos.y = 0
        self.Init_pos.z = 0

        self.des = 0

        self.desired_angle = 0


        self.flag = "1"


        self.cmd_twist = Twist()

        self.cmd_twist.linear.x = 0
        self.cmd_twist.linear.y = 0
        self.cmd_twist.linear.z = 0
        self.cmd_twist.angular.x = 0
        self.cmd_twist.angular.y = 0
        self.cmd_twist.angular.z = 0 
    
    def lidar_callback(self, lindata):

        self.varsaw=lindata.x

        # print ("Range:", self.varsaw)
        # print ("Angle:", self.localangle)

    def ang_callback(self,corddata):
        #print ("Pixel Cordinate of ball: %s", corddata.x)
        
        self.vars = corddata.x

    def predic_callback(self,predicdata):

        self.d.append(predicdata.x)
        self.d.popleft() 

        # print (self.d)

        # self.prediction = predicdata.x
        self.prediction = max(self.d,key=self.d.count)
        print ('prediction', self.prediction)


    def correct_angle(self,des_ang):

        if des_ang > math.pi:
            des_ang = des_ang - (2*math.pi)
        if des_ang <-math.pi:
            des_ang = des_ang + (2*math.pi)

        ib = des_ang - (math.pi/2)
        ic = des_ang + (math.pi/2)
        ie = des_ang - math.pi
        ig = des_ang + math.pi
        ik = des_ang

        if des_ang > math.pi:
            des_ang = des_ang - (2*math.pi)
        if des_ang <-math.pi:
            des_ang = des_ang + (2*math.pi)


        if des_ang > 0:
            min_des = min(abs(ib),abs(ie),abs(ik))

        if des_ang < 0:
            min_des = min(abs(ic),abs(ig),abs(ik))

        if min_des == abs(ik):
            des_ang = 0

        if min_des == abs(ib):
            des_ang = math.pi/2

        if min_des == abs(ic):
            des_ang = -math.pi/2

        if min_des == abs(ie):
            des_ang = math.pi

        if min_des == abs(ig):
            des_ang = -math.pi

        return des_ang


    def odom_callback(self, odom):

        self.update_Odometry(odom)

        if self.flag == "1":

            # print ('ang_error',self.vars)

            # print ('lidar_reading',self.varsaw)
            
            # print ('prediction',self.prediction)

            if self.vars:

                ang_error = 0 - self.vars

                lin_error =  self.varsaw - 0.4

                print ('ang_error',ang_error)

                print ('lidar_reading',lin_error)

                # if abs(ang_error) > 5 or abs(lin_error) > 0.05:    

                #     ang_vel = 0.007 * ang_error
                #     lin_vel = 0.1 * lin_error
                #     self.cmd_twist.angular.z = ang_vel
                #     self.cmd_twist.linear.x = lin_vel

                # else: 


                #     self.cmd_twist.linear.x = 0
                #     self.cmd_twist.angular.z = 0

                #     self.flag = False

                #     if self.prediction == 1:
                #         self.des = (math.pi/2) + self.globalAng
                #         self.desired_angle = self.correct_angle(self.des)                        

                #     if self.prediction == 2:
                #         self.des = (-math.pi/2) + self.globalAng
                #         self.desired_angle = self.correct_angle(self.des)    

                #     if self.prediction == 3:
                #         self.des = (math.pi) + self.globalAng
                #         self.desired_angle = self.correct_angle(self.des)  

                #     if self.prediction == 4:
                #         self.des = (math.pi) + self.globalAng
                #         self.desired_angle = self.correct_angle(self.des)

                #     if self.prediction == 5:
                #         self.desired_angle = self.globalAng

                if abs(ang_error) > 5:    

                    ang_vel = 0.007 * ang_error
                    self.cmd_twist.angular.z = ang_vel

                else: 

                    self.cmd_twist.angular.z = 0

                    if abs(lin_error) > 0.05:

                        lin_vel = 0.2 * lin_error
                        self.cmd_twist.linear.x = lin_vel

                    else:
                        self.cmd_twist.linear.x = 0                
                        self.flag = "2"

                        if self.prediction == 0:
                            self.des = (math.pi/2) + self.globalAng
                            self.desired_angle = self.correct_angle(self.des)                        

                        if self.prediction == 1:
                            self.des = (math.pi/2) + self.globalAng
                            self.desired_angle = self.correct_angle(self.des)                        

                        if self.prediction == 2:
                            self.des = (-math.pi/2) + self.globalAng
                            self.desired_angle = self.correct_angle(self.des)    

                        if self.prediction == 3:
                            self.des = (math.pi) + self.globalAng
                            self.desired_angle = self.correct_angle(self.des)  

                        if self.prediction == 4:
                            self.des = (math.pi) + self.globalAng
                            self.desired_angle = self.correct_angle(self.des)

                        if self.prediction == 5:
                            self.flag ="3"                    

                        print ('actual_angle',self.des*180/math.pi)
                        print ('desired_angle',self.desired_angle*180/math.pi)


        if self.flag == "2":

            self.cmd_twist.linear.x = 0

            ang_error1 = self.desired_angle -  self.globalAng

            if ang_error1 > math.pi:
                ang_error1 = ang_error1 - (2*math.pi)
            if ang_error1 <-math.pi:
                ang_error1 = ang_error1 + (2*math.pi)

            if abs(ang_error1) > 0.05:
                ang_vel1 = 0.4 * ang_error1
                self.cmd_twist.angular.z = ang_vel1
            else:
                self.flag = "1"
                self.cmd_twist.angular.z = 0 

        if self.flag == "3":  
            self.cmd_twist.linear.x = 0
            self.cmd_twist.angular.z = 0 


        self.publisher.publish(self.cmd_twist)


    def update_Odometry(self,Odom):
        
        position = Odom.pose.pose.position

        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

        if self.Init:
            #The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
            self.Init_pos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Init_pos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Init_pos.z = position.z

        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        

        #We subtract the initial values
        self.globalPos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos.x
        self.globalPos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos.y
        self.globalAng = orientation - self.Init_ang

        self.globalAng = np.arctan2(np.sin(self.globalAng),np.cos(self.globalAng))

def main():
    rospy.init_node('go_To_Goal', anonymous=True)
    process()
    try:
        rospy.spin()
    except KeyboardInterrupt:
       print("Shutting down ROS")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
