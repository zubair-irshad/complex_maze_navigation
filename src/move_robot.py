#!/usr/bin/env python

import roslib
import rospy
import numpy as np
import cv2
import sys, time


from sensor_msgs.msg import CompressedImage

from geometry_msgs.msg import Twist

from geometry_msgs.msg import Point
   

class chase:

	def __init__(self):

		global last_time

		global iteration_time

		last_time = 0

		iteration_time = 0

		integral = 0


		rospy.init_node('move_robot', anonymous=True)

		self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=3)

		rospy.Subscriber("/ang_cord",Point,self.ang_callback, queue_size=1)

		rospy.Subscriber("/lin_cord", Point, self.lin_callback, queue_size=1)

		rospy.Subscriber("/predic", Point, self.predic_callback, queue_size=1)

		self.vars = 0

		self.varsaw = 0

		self.prediction = 0

		# while self.diff_x is None:
		# 	pass

		# while self.varsaw is None:
		# 	pass

		# self.varsaw = 0

		#print ("diff_x",self.diff_x)

		#print ("diff_x",self.varsaw)	

		rate = rospy.Rate(5)		
		
		cmd_twist= Twist()
		cmd_twist.linear.x = 0;
		cmd_twist.linear.y = 0;
		cmd_twist.linear.z = 0;
		cmd_twist.angular.x = 0;
		cmd_twist.angular.y = 0; 	
		cmd_twist.angular.z = 0;

		p_gain = 0.05
		i_gain = 0.0001
		d_gain = 0

		ang_tolerance = 0.01
		derivatrive = 0
		ang_error_last = 0

		ang_error = 0

		p_gain2 = 0.2
		
		lin_tolerance = 0.3

		#self.vars = self.diff_x/self.varsaw

		#print ("diff_x",self.vars)

		while not rospy.is_shutdown():


			# + i_gain*integral + d_gain*derivative


			if self.vars:

				ang_error = 0 - self.vars

				# current_time = time.time()

				# iteration_time = current_time - last_time

				#print ("time",iteration_time)

				# if last_time == 0:
				# 	pass
				# else:
				# 	integral = integral + (ang_error*iteration_time)

				# print ("integral",integral)
				# print ("it time",iteration_time)
				# derivative = (ang_error - ang_error_last)/iteration_time

				# angular_z_vel = (p_gain * ang_error) + (i_gain * integral)	

				angular_z_vel = (p_gain * ang_error) 

				cmd_twist.linear.x = 0;
				cmd_twist.linear.y = 0;
				cmd_twist.linear.z = 0;
				cmd_twist.angular.x = 0;
				cmd_twist.angular.y = 0; 	
				cmd_twist.angular.z = angular_z_vel; 
				self.pub.publish(cmd_twist)
				rate.sleep()



				lin_error =  self.varsaw - 0.3

				lin_vel = p_gain2 * lin_error

					#print ("linerror:",lin_error)
				cmd_twist.linear.x = lin_vel;
				cmd_twist.linear.y = 0;
				cmd_twist.linear.z = 0;
				cmd_twist.angular.x = 0;
				cmd_twist.angular.y = 0; 	
				cmd_twist.angular.z = 0; 
				self.pub.publish(cmd_twist)
				rate.sleep()

			else: 
				cmd_twist.linear.x = 0;
				cmd_twist.linear.y = 0;
				cmd_twist.linear.z = 0;
				cmd_twist.angular.x = 0;
				cmd_twist.angular.y = 0; 	
				cmd_twist.angular.z = 0;
				self.pub.publish(cmd_twist)
				rate.sleep()

			last_time = current_time

		cmd_twist.linear.x = 0
		cmd_twist.angular.z = 0

		pub.publish(cmd_twist)

		rospy.spin()


	def ang_callback(self,corddata):
		#print ("Pixel Cordinate of ball: %s", corddata.x)
		
		self.vars = corddata.x
		#print ("diff_x",self.vars)

	def lin_callback(self,lindata):
		self.varsaw = lindata.x

	def predic_callback(self,predicdata):
		self.prediction = lindata.x


  
if __name__ == '__main__':
	try:
		chase()
	except KeyboardInterrupt:
		print "Shutting down ROS Image feature detector module"

