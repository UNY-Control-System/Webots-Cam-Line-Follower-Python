from controller import Robot, Camera, DistanceSensor
import cv2
import numpy as np

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice('camera')
camera.enable(timestep)

# Motors - set to default (velocity = 0)
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# Set the previous error and integral for PID to null
e_prev, angle = None, None
integral, PID, delta_err, adj_speed = 0, 0, 0, 0

# Scale is used for OpenCV Resize Window
scale = 0.35

# Steer PID Configuration
def SteerPID(error, integral, derivative):
    Kp = 0.03
    Ki = 0.0000001
    Kd = 0.00001
    u = Kp * error + Ki * integral + Kd * derivative
    return u

# Speed PID Configuration
def SpeedPID(error, integral, derivative):
    Kp = 0.1
    Ki = 0.000001
    Kd = 0.002
    u = Kp * error + Ki * integral + Kd * derivative
    return u

print("Left Motor, Right Motor, Steer, Base Speed, Error, Delta Error, Angle")

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Access to the camera.
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

    # Mask the image
    mask = np.zeros(gray.shape, np.uint8)
    mask[225:650, 0:1280] = 255
    masked_black_mask = cv2.bitwise_or(black_mask, black_mask, mask=mask)
    
    # Draw the set point circle
    x_pot = int(width/2)
    y_pot = int(height*0.95)
    cv2.circle(img_np, (x_pot, y_pot), 10, (0, 0, 255), -1)

    # Find the contours of the black mask
    contours, _ = cv2.findContours(masked_black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Get the biggest contour
        biggest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(biggest_contour)

        # Draw the bounding rectangle
        # Draw a cannny from contour and get the center
        cv2.drawContours(img_np, [biggest_contour], -1, (0, 255, 0), 2)

        # Get the center of the bounding rectangle
        center_x = x + w//2
        center_y = y + h//2

        # Draw the center of the bounding rectangle
        cv2.circle(img_np, (center_x, center_y), 5, (255, 0, 0), -1)

        # Draw the line from the center of the bounding rectangle to the center of the image
        cv2.line(img_np, (center_x, center_y), (x_pot, y_pot), (255, 255, 0), 2)
        
        # Calculate the PID
        # Get an error from angle using degrees
        angle = np.arctan2(center_y - y_pot, center_x - x_pot)
        angle = np.degrees(angle)
        error = 90 + angle

        # Calculate the integral
        integral = integral + error

        # Get the previous error first to calculate the PID then calculate the PID
        if e_prev is not None:
            # Calculate the derivative
            derivative = error - e_prev
            PID = SteerPID(error, integral, derivative)
            adj_speed = SpeedPID(error, integral, derivative)

            # Get the delta error
            delta_err = e_prev - error
        
        # Set the current as previous error to next iteration
        e_prev = error

    # Exit the OpenCV Windows using 'q' key on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Set the motors velocity
    leftVel = 6.5 - abs(adj_speed) + PID
    rightVel = 6.5 - abs(adj_speed) - PID

    # Limit the motor velocity
    if leftVel >= 7.25:
        leftVel = 7.25
    elif rightVel >= 7.25:
        rightVel = 7.25
    if leftVel < 0:
        leftVel = 0
    elif rightVel < 0:
        rightVel = 0
    
    print(f"{leftVel}, {rightVel}, {PID}, {adj_speed}, {error}, {delta_err}, {angle}")
    # print(f"LM: {leftVel}, RM: {rightVel}, Steer: {PID}, BS: {adj_speed}, Err: {error}, DE: {delta_err}, Ang: {angle}")
    
    # Set motors velocity
    leftMotor.setVelocity(leftVel)
    rightMotor.setVelocity(rightVel)

    # Show the OpenCV Window
    # img_np_resized = cv2.resize(img_np, (int(img_np.shape[1] * scale), int(img_np.shape[0] * scale)))
    # cv2.imshow("Webots Camera", img_np_resized)

# Close OpenCV Window when the robot restart
cv2.destroyAllWindows()