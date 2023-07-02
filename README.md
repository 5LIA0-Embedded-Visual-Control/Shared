# BRAIN

We will implement a ROS node named "brain" to assemble multiple functions, such as finding duckies or house, going towards the house, obstacle avoidance.

## How to Use It

### 0. Do you get the physical Duckiebot ?

Please say yes.

If not, go and buy one. 

### 1. git clone this repository to a folder

Basic operation

### 2. type two commands in terminal

Open a terminal in the folder saving all these files

Type "dts devel build -f -H db1.local" at first -- build an image 

And then, type "dts devel run -H db1.local"  -- run the image

### 3. just wait

you can observe the state of duckiebot by looking at lines shown in terminal.

Loading yolov5 model will take some time, be patient

## Finite State Machine
file:///home/laixunqin/Downloads/FSM(1).png

### FindDuck
Spin for 6 frames out of 10, stop for 4 frames out of 10, instead of spinning all the time.
Inference result would be worse when the duckiebot keep spinning

### TowardsDuck
Duckiebot needs to avoid the house when it is arriving at the position of a duck.
We implement a PID controller to control duckiebot to go towards the duck.
Sometimes there isn't any duck in the view of duckiebot because of the sway, and it need to degenerate to "FindDuck".
If there is two duck in the view, the duckiebot is able to determine which duck should be followed.(function: WhichDuck)

### FindHouse
Same as FindDuck

### TowardsHouse
Duckiebot needs to avoid another duck(if so) when it is arriving at the position of a house.
We have implemented a function to decide which duck should be avoided, because the duck which is carried by duckiebot is also in the view.
Sometimes there isn't any house in the view of duckiebot because of the sway, and it need to degenerate to "FindHouse".

### Avoidance
Duckiebot will change its  direction to avoid the obstacle which is closed to duckiebot.

### Backward
Duckiebot will go backward after it successfully carried a duck to his home, which leaves some space for next search.

## Function

### FindIndex(self, lst)
Get all indexes of duck in the list.

### JudgeDirection(self, x)
Determine the direction, left or right, to avoid the duck

### WhichDuck(self, obj_lst, box_lst)
Determine which duck should be followed when there are two duck in the view

### FilterDuck(self, obj_lst, box_lst)
Enable the duckiebot to know which duck should be avoided when it is carrying one duck to the house

### PIDtowardObject(self,xt,yt)
Enable the duckiebot to go to the destination



