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

# Motors - set to default (velocity = 0)
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# PID Configuration
# prev_angle = 0
# angle = 0
e_prev = 0
integral = 0

scale = 0.65

def CalculatePID(error, integral, derivative):
    Kp = 0.02
    Ki = 0.00000001
    Kd = 0.00001
    u = Kp * error + Ki * integral + Kd * derivative
    return u

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

    # Change to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]
    gray_resized = cv2.resize(gray, (int(img_np.shape[1] * scale), int(img_np.shape[0] * scale)))

    # Filter the color
    _, black_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    black_mask_resized = cv2.resize(black_mask, (int(img_np.shape[1] * scale), int(img_np.shape[0] * scale)))

    # Draw the set point circle
    x_pot = int(width/2)
    y_pot = int(height/2)
    cv2.circle(img_np, (x_pot, y_pot), 10, (0, 0, 255), -1)

    # Find the contours of the black mask
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Get the biggest contour
        biggest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(biggest_contour)

        # Draw the bounding rectangle
        cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Get the center of the bounding rectangle
        center_x = x + w//2
        center_y = y_pot

        # Draw the center of the bounding rectangle
        cv2.circle(img_np, (center_x, center_y), 5, (255, 0, 0), -1)

        # Draw the line from the center of the bounding rectangle to the center of the image
        cv2.line(img_np, (center_x, center_y), (x_pot, y_pot), (255, 255, 0), 2)

        # Calculate the angle
        # angle = np.arctan2(center_y - y_pot, center_x - x_pot)
        # angle = np.degrees(angle)
        # I still confused. Should I use angle instead of using delta x or not.
        
        # Calculate the PID
        time = robot.getTime()
        error = x_pot - center_x
        integral = integral + error
        derivative = error - e_prev

        if time == 0.032:
            PID = 0

        else:
            # PID = CalculatePID(x_pot, center_x, t_prev, t)
            PID = CalculatePID(error, integral, derivative)
            # print(f"PID: {PID}, Angle: {angle}")

        delta_err = e_prev - (error)
        e_prev = error

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print(f"{x_pot - center_x}, {delta_err}")

    # Set the motors velocity
    leftVel = 3.5 - PID
    rightVel = 3.5 + PID
    # print(f"Left Velocity: {leftVel}, Right Velocity: {rightVel}, PID: {PID}, Error: {error}")

    # Set motors velocity
    leftMotor.setVelocity(leftVel)
    rightMotor.setVelocity(rightVel)

    img_np_resized = cv2.resize(img_np, (int(img_np.shape[1] * scale), int(img_np.shape[0] * scale)))
    cv2.imshow("Gray", gray_resized)
    cv2.imshow("Black Mask", black_mask_resized)

    cv2.imshow("Webots Camera", img_np_resized)

cv2.destroyAllWindows()

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)

# Enter here exit cleanup code.
