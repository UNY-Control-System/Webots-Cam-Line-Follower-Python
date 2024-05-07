"""camera_access controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera, DistanceSensor
import cv2
import numpy as np

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)
camera = robot.getDevice('camera')
camera.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.
    image = camera.getImage()
    img_np = np.frombuffer(image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    img_np = img_np[:, :, :3]
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_np_resize = cv2.resize(img_np, (int(img_np.shape[1] * 0.3), int(img_np.shape[0] * 0.3)))
    cv2.imshow("Webots Camera", img_np_resize)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)

# Enter here exit cleanup code.
