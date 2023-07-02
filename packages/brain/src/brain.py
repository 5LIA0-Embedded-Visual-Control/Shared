#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import time

from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Twist2DStamped, LEDPattern
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import ColorRGBA

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

        self.pub_detections_image = rospy.Publisher(
            "~image/compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.DEBUG
        )

        led_topic = f"/{self.veh}/led_emitter_node/led_pattern"
        self.led_pub = rospy.Publisher(
            led_topic, 
            LEDPattern, 
            queue_size=1,
            dt_topic_type=TopicType.DRIVER
        )

        # Construct subscribers, receive image from camera
        self.sub_image = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.brain,
            buff_size=10000000,
            queue_size=1,
        )
        
        self.spin = 0

        self.backward = 0

        self.idealx = 208
        self.idealy = 300
        self.sum_error = 0
        self.last_error = 0
        self.Kp = 0.5
        self.Ki = 0
        self.Kd = 0
        
        self.bridge = CvBridge()
        self.model_wrapper = Wrapper() # Load the model
        
        self.state = "FindDuck"  # Intial
        
        self.pre = ''
        self.avoid = 0
        self.direction = 0
        
        self.initialized = True
        self.log("Initialized!")
    
    def SendCommand(self, header, v, omega):
        control_msg = Twist2DStamped()
        control_msg.header = header
        control_msg.v = v
        control_msg.omega = omega
        self.pub_car_cmd.publish(control_msg)
    
    def ShowResults(self, classes, bboxes, rgb):
        colors = {0: (0, 0, 255), 1: (255, 0, 0)}
        names = {0: "duckie", 1: "house"}
        font = cv2.FONT_HERSHEY_SIMPLEX
        for clas, box in zip(classes, bboxes):
            pt1 = np.array([int(box[0]), int(box[1])])
            pt2 = np.array([int(box[2]), int(box[3])])
            pt1 = tuple(pt1)
            pt2 = tuple(pt2)
            color = tuple(reversed(colors[clas]))
            name = names[clas]
            # draw bounding box
            rgb = cv2.rectangle(rgb, pt1, pt2, color, 2)
            # label location
            text_location = (pt1[0], min(pt2[1] + 30, IMAGE_SIZE))
            # draw label underneath the bounding box
            rgb = cv2.putText(rgb, name, text_location, font, 1, color, thickness=2)

        bgr = rgb[..., ::-1]
        obj_det_img = self.bridge.cv2_to_compressed_imgmsg(bgr)
        self.pub_detections_image.publish(obj_det_img)

    def FindIndex(self, lst):
        return [index for index, element in enumerate(lst) if element == 0]

    def JudgeDirection(self, x):
        if x > 208:
            return 1
        else:
            return -1
        
    def WhichDuck(self, obj_lst, box_lst):
        idxs = self.FindIndex(obj_lst)
        num = len(idxs)

        if num == 1:
            return idxs[0]
        
        else: # num == 2
            idx1 = idxs[0]
            idx2 = idxs[1]
            duck1x = (box_lst[idx1][0] + box_lst[idx1][2])/2
            duck1y = (box_lst[idx1][1] + box_lst[idx1][3])/2
            duck2x = (box_lst[idx2][0] + box_lst[idx2][2])/2
            duck2y = (box_lst[idx2][1] + box_lst[idx2][3])/2

            if abs(duck1y - duck2y) > 10:
                return idx2 if duck2y > duck1y else idx1
            
            else:
                return idx2 if duck2x < duck1x else idx1
    
    def FilterDuck(self, obj_lst, box_lst):
        idxs = self.FindIndex(obj_lst)
        idx1 = idxs[0]
        idx2 = idxs[1]
        duck1x = (box_lst[idx1][0] + box_lst[idx1][2])/2
        duck1y = (box_lst[idx1][1] + box_lst[idx1][3])/2
        duck2x = (box_lst[idx2][0] + box_lst[idx2][2])/2
        duck2y = (box_lst[idx2][1] + box_lst[idx2][3])/2
        dist1 = (duck1x - self.idealx) ** 2 + (duck1y - self.idealy) ** 2
        dist2 = (duck2x - self.idealx) ** 2 + (duck2y - self.idealy) ** 2

        return idx1 if dist1 > dist2 else idx2
    

    def PIDtowardsObject(self, xt, yt):
        error_x = self.idealx - xt
        error_y = abs(self.idealy - yt)

        error = error_x/error_y
        self.sum_error += error
        self.last_error = error

        omega = self.Kp * error + self.Ki * self.sum_error + self.Kd * (self.last_error - error)

        if abs(omega) > 2:
            omega = np.sign(omega) * 2

        return omega
    

    def brain(self, image_msg):
        # wait for initialization, it takes some time
        if not self.initialized:
            print("wait until initialized")
            return
        
        print("Brain is working~")
        print("Current state: ", self.state)

        # Avoid the obstacle
        if self.state == "Avoidance":
            if self.avoid < 15:
                self.SendCommand(image_msg.header, 0.4, self.direction * 4)
                self.avoid += 1
            
            elif self.avoid < 30:
                self.SendCommand(image_msg.header, 0.4, -self.direction * 4)
                self.avoid += 1
            
            if self.avoid == 30:
                self.avoid = 0
                self.state = "TowardsHouse" if self.pre == "TowardsHouse" else "TowardsDuck"
            
            return
        
        # Backward after the duckiebot successfully send duck home
        if self.state == "Backward":

            if self.backward < 50:
                self.SendCommand(image_msg.header, -0.4, 0)
                self.backward += 1
            
            else:
                self.backward = 0
                self.state = "FindDuck"
            
            return

        
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
        print(classes)
        print(bboxes)

        self.ShowResults(classes, bboxes, rgb)

        # Try to find a duck
        if self.state == "FindDuck":
            # Cannot see the duck
            if 0 not in classes:
                self.spin += 1
                if (self.spin % 10) < 5:
                    self.SendCommand(image_msg.header, 0, 4)
                    return
                
                self.SendCommand(image_msg.header, 0, 0)
                print("Where is Asian Duck?")
            
            # Already saw the duck
            else:
                self.spin = 0
                self.SendCommand(image_msg.header, 0, 0.0)
                self.state = "TowardsDuck"
                print("Oh! It's a Duck!")
        
        # Head to duck
        if self.state == "TowardsDuck":
            # Make sure see the duck
            if 0 not in classes:
                self.state = "FindDuck"
                return
            
            # Avoid the house
            if 1 in classes:
                idx_house = classes.index(1)
                obs_x1 = bboxes[idx_house][0]
                obs_x2 = bboxes[idx_house][2]
                obs_y = bboxes[idx_house][3]

                cond1 = 140 < obs_x1 and obs_x1 < 340 
                cond2 = 140 < obs_x2 and obs_x2 < 340
                
                if (cond1 or cond2) and obs_y > 240:
                    self.direction = 1 if cond1 == True else -1
                    self.SendCommand(image_msg.header, 0, 0.0)
                    self.state = "Avoidance"
                    self.pre = "TowardsDuck"
                    return

            idx  = self.WhichDuck(classes, bboxes)

            # duck position
            duck_x = (bboxes[idx][0] + bboxes[idx][2])/2
            duck_y = (bboxes[idx][1] + bboxes[idx][3])/2

            # Time to stop
            if (duck_y > 360 and 135 < duck_x and duck_x < 314):
                self.SendCommand(image_msg.header, 0, 0.0)
                print("Hey! I got you, my boy!")
                self.sum_error = 0
                self.last_error = 0
                self.state = "FindHouse"
                return

            self.SendCommand(image_msg.header, 0.05, self.PIDtowardsObject(duck_x, duck_y))
            print("On my way to the duck!")

        
        # Already pick up a Duck, try to find Green House
        if self.state == "FindHouse":
            # Cannot see the house
            if 1 not in classes:
                self.spin += 1
                if (self.spin % 10) < 5:
                    self.SendCommand(image_msg.header, 0, 4)
                    return
                
                self.SendCommand(image_msg.header, 0, 0)
                print("Where is my house?")
            # Already saw the house
            else:
                self.spin = 0
                self.SendCommand(image_msg.header, 0, 0)
                self.state = "TowardsHouse"
                print("Oh! It's my house!")
        
        # Carry the duck and head to house
        if self.state == "TowardsHouse":
            # Make sure see the house
            if 1 not in classes:
                self.state = "FindHouse"
                return
            
            num_duck = classes.count(0)

            # Avoid another duck
            if num_duck == 2:
                idx_duck = self.FilterDuck(classes, bboxes)
                duck_x = (bboxes[idx_duck][0] + bboxes[idx_duck][2])/2
                duck_y = bboxes[idx_duck][3]

                if 140 < duck_x and duck_x < 340 and duck_y > 220:
                    self.direction = self.JudgeDirection(duck_x)
                    self.SendCommand(image_msg.header, 0, 0.0)
                    self.state = "Avoidance"
                    self.pre = "TowardsHouse"
                    return

            idx  = classes.index(1)

            house_x = (bboxes[idx][0] + bboxes[idx][2])/2
            house_y = bboxes[idx][3]

            # Time to stop:
            if house_y > 300 and 135 < house_x and house_x < 314:
                self.SendCommand(image_msg.header, 0, 0.0)
                print("Hey! Please get off my car!")
                self.sum_error = 0
                self.last_error = 0
                self.state = "Backward"
                time.sleep(5)  
                return

            self.SendCommand(image_msg.header, 0.05, self.PIDtowardsObject(house_x, house_y))
            print("On my way to house!")


if __name__ == "__main__":
    # Initialize the node
    brain = brain(node_name="brain")
    
    # Keep it spinning
    rospy.spin()
