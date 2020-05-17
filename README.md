Complex Maze Navigation using Image Classification
---------------------------------------------------
ROS based turtlebot3 mobile robot navigator using sign recognition based on image classification. 

ROS package 'navigate_to_goal' contains five nodes. 

- Train_classifier node:
This node should be run the first time to train the classifier. This will save the classifier as a .pkl file and will be used later on to predict sign values by the go_to_Goal node.

- go_to_Goal Node:

This node does two important tasks:

1) Following the sign:
Firstly, it follows the sign by adjusting its angualr position first and then the linear position by keeping the robot 3 m from the sign. It automatically selects the sign to follow based on the which sign is closest to the center of the screen. 

2) Predicting the sign and navigation
Secondly, when the robot is 3 m away from the sign, it reads classifier from classifier.pkl file and predicts the value based on the training data set it has already been trained on. It then makes a maneuver based on the output value. For instance 90 degrees clockwise for a right turn. 

The other two nodes are additional nodes which could be explored alternatively to carry out the same task.

Cheers !


