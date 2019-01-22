# navigate_to_goal
ROS based turtlebot3 mobile robot navigator using sign recognition based on image classification. 

ROS package 'navigate_to_goal' contains five nodes. 

- Train_classifier node should be run the first time to train the classifier. This will save the classifier as a .pkl file and will be used later on to predict sign values by the go_to_Goal node.
- 
