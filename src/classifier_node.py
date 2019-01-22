#!/usr/bin/env python

# ROS
from irshad_final.srv import Classify
import numpy as np
import rospy
import rospkg

# scikit-learn
from sklearn.externals import joblib





class ClassifierNode:

    def __init__(self):
        """Initialize classifier service in a ROS node."""
        # Load the previously-trained classifier model

        filepath = '/home/mirshad/catkin_ws/src/irshad_final/data/classifier/classifer3.pkl'
        # filepath = rospy.get_param('~file_name', 'classifier1.pkl')
        # if len(filepath) > 0 and filepath[0] != '/':
        #     filepath = rospkg.RosPack().get_path('image_classifier') + '/data/classifier/' + filepath
        self.model = joblib.load(filepath)

        self.service = rospy.Service('~classify', Classify, self.classify)

    def classify(self, req):
        """Return binary classification of an ordered grasp pair feature vector."""
        return self.model.predict(np.asarray(req.feature_vector).reshape(1, -1))

if __name__ == '__main__':
    rospy.init_node('classifier_node')

    classifier_node = ClassifierNode()
    print 'Classifier node initialized.'

    rospy.spin()
