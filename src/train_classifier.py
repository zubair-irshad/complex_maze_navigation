#!/usr/bin/env python

# ROS
import rospy
import rospkg

# numpy
import numpy


import pickle
# scikit-learn (some common classifiers are included as examples, feel free to add your own)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC




def train_classifier():
    """Train a classifier and save it as a .pkl for later use."""
    rospy.init_node('train_classifier')

    filepath = '/home/mirshad/Downloads/Irshad_MuhammadZubair_orig/foo.csv'

    # filepath = rospy.get_param('~file_name', 'foo.csv')
    # if len(filepath) > 0 and filepath[0] != '/':
    #     rospack = rospkg.RosPack()
    #     filepath = rospack.get_path('image_classifier') + '/data/training/' + filepath

    split = 0.4

    data, label = load_data(filepath)
    print '\nImported', data.shape[0], 'training instances'

    data, label = shuffle(data, label)
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=split)

    print 'Training classifier...'

    ##########################################################################
    # Begin classifier initialization code (You write this!)

    # Initialize the classifier you want to train with the parameters you want here:
    # Install scikit-learn with these instructions: http://scikit-learn.org/stable/install.html
    # Models, documentation, instructions, and examples can be found here:
    #   http://scikit-learn.org/stable/supervised_learning.html#supervised-learning

    classifier = LinearSVC()  # TODO: Replace this with the classifier you want

    # End image processing code (You write this!)
    ############################################################################################

    classifier.fit(data_train, label_train)

    print('Detailed results on a %0.0f/%0.0f train/test split:' % ((1 - split)*100, split*100))
    predicted = classifier.predict(data_test)


    print("test acuracy", accuracy_score(label_test, predicted))


    # print(metrics.classification_report(label_test, predicted))
    # print(metrics.confusion_matrix(label_test, predicted))

    print('Training and saving a model on the full dataset...')
    classifier.fit(data, label)


    # pkl_filename = "/home/mirshad/catkin_ws/src/irshad_final/data/classifier/classifier.pkl"  

    # with open(pkl_filename, 'wb') as file:  
    #     pickle.dump(classifier, file)

    joblib.dump(classifier, '/home/mirshad/catkin_ws/src/irshad_final/data/classifier/classifier3.pkl')
    print('Saved model classifier.pkl to current directory.')


def load_data(filepath):
    """Parse training data and labels from a .csv file."""
    data = numpy.loadtxt(filepath, delimiter=',')
    x = data[:, :data.shape[1] - 1]
    y = data[:, data.shape[1] - 1]
    return x, y


if __name__ == '__main__':
    try:
        train_classifier()
    except rospy.ROSInterruptException:
        pass
