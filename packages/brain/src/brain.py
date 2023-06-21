#!/usr/bin/env python3

import cv2
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Twist2DStamped, EpisodeStart
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

from nn_model.constants import IMAGE_SIZE
from nn_model.model import Wrapper

class brain(DTROS):
    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(brain, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.initialized = False
        self.log("Initializing!")

        self.veh = rospy.get_namespace().strip("/")

        # Construct publishers, send control command
        car_cmd_topic = f"/{self.veh}/joy_mapper_node/car_cmd"
        self.pub_car_cmd = rospy.Publisher(
            car_cmd_topic,
            Twist2DStamped,
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )

        # Construct subscribers, receive image from camera
        self.sub_image = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.brain,
            buff_size=10000000,
            queue_size=1,
        )
        
        self.bridge = CvBridge()
        self.model_wrapper = Wrapper() # Load the model
        self.state = "FindHouse"  # Intial 
        self.initialized = True
        self.log("Initialized!")
    

    def brain(self, image_msg):
        # wait for initialization, it takes some time
        if not self.initialized:
            print("wait until initialized")
            return
        
        print("Brain is working~")

        # Decode from compressed image with OpenCV
        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr("Could not decode image: %s" % e)
            return

        # from bgr to rgb and resize the image
        rgb = bgr[..., ::-1]
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))

        # Yolov5 inference
        bboxes, classes, scores = self.model_wrapper.predict(rgb)
        print(bboxes)
        
        # defination of control message
        car_control_msg = Twist2DStamped()
        car_control_msg.header = image_msg.header

        if self.state == "FindAsianDuck":
            # Cannot see Asian Duck
            if 0 not in classes:
                car_control_msg.v = 0
                car_control_msg.omega = 0.4
                self.pub_car_cmd.publish(car_control_msg)
                print("Where is Asian Duck?")
            # Already saw Asian Duck
            else:
                car_control_msg.v = 0
                car_control_msg.omega = 0.0
                self.pub_car_cmd.publish(car_control_msg)
                print("Oh! It's Asian Duck")

        if self.state == "TowardsAsianDuck":
            pass

        # Already pick up Asian Duck, try to find Green House
        if self.state == "FindHouse":
            # Cannot see the house
            if 1 not in classes:
                car_control_msg.v = 0
                car_control_msg.omega = 0.4
                self.pub_car_cmd.publish(car_control_msg)
                print("Where is my house?")
            # Already saw the house
            else:
                car_control_msg.v = 0
                car_control_msg.omega = 0.0
                self.pub_car_cmd.publish(car_control_msg)
                print("Oh! It's my house!")
        
        if self.state == "TowardsHouse":
            pass
        
        if self.state == "FindAfricaDuck":
            # Cannot see Africa Duck
            if 2 not in classes:
                car_control_msg.v = 0
                car_control_msg.omega = 0.4
                self.pub_car_cmd.publish(car_control_msg)
                print("Where is Africa Duck?")
            # Already saw Africa Duck
            else:
                car_control_msg.v = 0
                car_control_msg.omega = 0.0
                self.pub_car_cmd.publish(car_control_msg)
                print("Oh! It's Africa Duck")
        

        
if __name__ == "__main__":
    # Initialize the node
    brain = brain(node_name="brain")
    
    # Keep it spinning
    rospy.spin()
