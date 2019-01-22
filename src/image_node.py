#!/usr/bin/env python

# ROS
import rospy

import cv2

import numpy as np

# from irshad_final.srv import Classify

from sensor_msgs.msg import CompressedImage

from cv_bridge import CvBridge

from sklearn.externals import joblib

from geometry_msgs.msg import Point

from sensor_msgs.msg import LaserScan


class ImageNode:

    def __init__(self):
        """Initialize image subscriber and classifier client."""

        self.bridge = CvBridge() #Bridge converts the image from ros to openCV
        self.blurSize = 9 

        self.x_error = Point()
        self.pt2 = Point()
        self.pt3 = Point()

        self.pub = rospy.Publisher('ang_cord', Point, queue_size=3)
        self.pub2 = rospy.Publisher('lin_cord', Point, queue_size=3)
        self.pub3 = rospy.Publisher('predic', Point, queue_size=3)

        self.image_subscriber = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.imageCallback)
        self.lidar_subscriber = rospy.Subscriber("/scan",LaserScan, self.LIDAR)



        # print 'Waiting for classifier service to come up...'
        # rospy.wait_for_service('/classifier_node/classify')
        # self.classify_client = rospy.ServiceProxy('/classifier_node/classify', Classify)

    def LIDAR(self,ldata):

        min_lid = 0.12
        max_lid = 3.5

        # minDist = min(list(filter(lambda x: x<max_lid and x>min_lid, ldata.ranges)))
        minDist = float("inf")
        minAngle = 0
        for (angle, dist) in enumerate(ldata.ranges):
            if dist > min_lid and dist < max_lid and (angle>355 or angle<5):
                if dist < minDist:
                    minDist = dist
                    minAngle = angle

        self.pt2.x = minDist
        self.pub2.publish(self.pt2)  

        # print ('lidar distance',minDist)
        # print('Lidar data: ', self.pt2.x)  


    def imageCallback(self, image):
        """Process an image and return the class of any sign in the image."""

        ############################################################################################
        # Begin image processing code (You write this!)

        imgBGR = self.bridge.compressed_imgmsg_to_cv2(image, "bgr8")
        imgBlur = cv2.GaussianBlur(imgBGR,(self.blurSize,self.blurSize),0)

        lower_red1 = np.array([0,50,10])
        upper_red1 = np.array([30,255,255])

        lower_red2 = np.array([160,50,20])
        upper_red2 = np.array([180,255,255])

        lower_green = np.array([37, 40, 25])
        upper_green = np.array([78, 255, 255]) 

        lower_blue = np.array([80,65,30]) 
        upper_blue = np.array([120,255,255]) 

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))

        hsv_gen = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_gen, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_gen, lower_red2, upper_red2)
        mask3 = cv2.inRange(hsv_gen, lower_green, upper_green)
        mask4 = cv2.inRange(hsv_gen, lower_blue, upper_blue)

        mfin = mask1 + mask2 + mask3 + mask4

        clos1 = cv2.morphologyEx(mfin, cv2.MORPH_CLOSE, kernel)
        # opening = cv2.morphologyEx(clos1, cv2.MORPH_OPEN, kernel)


        (image, contours, _) = cv2.findContours(clos1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # print (contours.shape)

        approx=None

        w = 90
        h = 90

        p=[]

        cnt = None

        areas = []

            
        if not contours:
            resized=imgBGR[0:80,0:80]

        k_min = 9999


        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c,0.001*cv2.arcLength(c,True),True)
            cv2.drawContours(imgBGR, [approx], -1, (0, 255, 0), 2)
            areas.append(cv2.contourArea(c))

            x1,y1,w1,h1 = cv2.boundingRect(c)

            # print x1,y1,w1,h1            
            if h1 > 20 and w1 >20:
                x_error = x1 + (w1/2) - (imgBGR.shape[1]/2)
                if abs(x_error) < abs(k_min):
                    k_min = x_error

        if areas:
            max_index = np.argmax(areas)
            cnt=contours[max_index]
                
        if contours:
            x,y,w,h = cv2.boundingRect(cnt)

            # print (x,y,w,h)

            if h >=25:
                new_img=imgBGR[y:y+h,x:x+w]
                resized = cv2.resize(new_img, (100,100), interpolation = cv2.INTER_AREA)
                # elif h>53 and h<55:
                #     new_img=abc[y:y+2*h,x:x+w]
                #     resized = cv2.resize(new_img, (100,100), interpolation = cv2.INTER_AREA)
            elif h<25:
                new_img=imgBGR[0:80,0:80]
                resized = cv2.resize(new_img, (100,100), interpolation = cv2.INTER_AREA)
        else:
            k_min = 0

        # print x,y,w,h

        cv2.imshow('image',imgBGR)
        cv2.waitKey(2)


        self.x_error.x = k_min

        self.pub.publish(self.x_error)  

        # print ('ang_error', self.x_error.x)


        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        train_data = gray.reshape(1, 100*100)
        feature_vector = train_data  # TODO: Fill this in with the features you extracted from the image

        feature_vector = feature_vector.astype(np.float32)

        # End image processing code
        ############################################################################################

        filepath1 = '/home/mirshad/catkin_ws/src/irshad_final/data/classifier/classifier3.pkl'
        # filepath = rospy.get_param('~file_name', 'classifier1.pkl')
        # if len(filepath) > 0 and filepath[0] != '/':
        #     filepath = rospkg.RosPack().get_path('image_classifier') + '/data/classifier/' + filepath
        model = joblib.load(filepath1)

        y_pred = model.predict(feature_vector)

        self.pt3.x = y_pred
        self.pub3.publish(self.pt3)  

        # print ('pred',self.pt3.x)

        # classification = self.classify_client(feature_vector)
        # print('Classified image as: ', y_pred[0])
        # print ('error',self.x_error.x)


if __name__ == '__main__':
    rospy.init_node('image_node')

    image_node = ImageNode()
    print 'Image node initialized.'
    rospy.spin()
