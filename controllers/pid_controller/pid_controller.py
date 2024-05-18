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

# Set the previous error and integral for PID to zero
e_prev = 0
integral = 0

# Scale is used for OpenCV Resize Window
scale = 0.35

# PID Configuration
def CalculatePID(error, integral, derivative):
    Kp = 0.0059002274
    Ki = 0.00000
    Kd = 0.0000
    u = Kp * error + Ki * integral + Kd * derivative
    return u

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
    black_mask_resized = cv2.resize(black_mask, (int(img_np.shape[1] * scale), int(img_np.shape[0] * scale)))

    # Mask the image
    mask = np.zeros(gray.shape, np.uint8)
    mask[0:250, 0:1280] = 255
    masked_black_mask = cv2.bitwise_or(black_mask, black_mask, mask=mask)
    masked_black_mask_resized = cv2.resize(masked_black_mask, (int(img_np.shape[1] * scale), int(img_np.shape[0] * scale)))
    
    # Draw the set point circle
    x_pot = int(width/2)
    y_pot = int(height*0.75)
    cv2.circle(img_np, (x_pot, y_pot), 10, (0, 0, 255), -1)

    # Find the contours of the black mask
    # contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(masked_black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Get the biggest contour
        biggest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(biggest_contour)

        # Draw the bounding rectangle
        # cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 0)
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
        # Get the world time
        time = robot.getTime()

        # Get an error (Set point of the camera - Center of bounding rectangle)
        # error = x_pot - center_x
        # Get an error from angle using degrees
        error = np.arctan2(center_x - x_pot, center_y - y_pot)
        error = np.degrees(error)
        error = 180 - error

        print(f"Error: {error}")

        # Calculate the integral
        integral = integral + error

        # Calculate the derivative
        derivative = error - e_prev

        # Get the previous error first to calculate the PID then calculate the PID
        PID = 0 if time == 0.032 else CalculatePID(error, integral, derivative)
        
        # Get the delta error
        delta_err = e_prev - error

        # Set the current as previous error to next iteration
        e_prev = error

    # Exit the OpenCV Windows using 'q' key on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Set the motors velocity
    # leftVel = 6 - PID
    # rightVel = 6 + PID

    base_speed = 6
    adjust_speed = base_speed - PID

    print(base_speed, adjust_speed)

    leftVel = adjust_speed + PID
    rightVel = adjust_speed - PID

    if leftVel >= 6.28:
        leftVel = 6.28
    elif rightVel >= 6.28:
        rightVel = 6.28
    if leftVel < 0:
        leftVel = 0
    elif rightVel < 0:
        rightVel = 0
    # print(f"Left Velocity: {leftVel}, Right Velocity: {rightVel}, PID: {PID}, Error: {error}, Delta Error: {delta_err}")
    print(f"{leftVel}, {rightVel}, {PID}, {error}, {delta_err}")
    
    # Set motors velocity
    leftMotor.setVelocity(leftVel)
    rightMotor.setVelocity(rightVel)

    # Show the OpenCV Window
    img_np_resized = cv2.resize(img_np, (int(img_np.shape[1] * scale), int(img_np.shape[0] * scale)))
    cv2.imshow("Gray", gray_resized)
    cv2.imshow("Black Mask", black_mask_resized)
    cv2.imshow("Masked Black Mask", masked_black_mask_resized)
    cv2.imshow("Webots Camera", img_np_resized)

# Close OpenCV Window when the robot restart
cv2.destroyAllWindows()