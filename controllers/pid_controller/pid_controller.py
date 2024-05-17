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

scale = 0.65

# PID Configuration
def CalculatePID(error, integral, derivative):
    Kp = 0.02
    Ki = 0.00000001
    Kd = 0.00001
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
        
        # Calculate the PID
        # Get the world time
        time = robot.getTime()

        # Get an error (Set Point - Center of bounding rectangle)
        error = x_pot - center_x

        # Calculate the integral
        integral = integral + error

        # Calculate the derivative
        derivative = error - e_prev

        # Get the previous error first to calculate the PID
        if time == 0.032:
            PID = 0
        else:
            PID = CalculatePID(error, integral, derivative)
        
        # Get the delta error
        delta_err = e_prev - (error)

        # Set the current as previous error to next iteration
        e_prev = error

    # Exit the OpenCV Windows using 'q' key on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Set the motors velocity
    leftVel = 3.5 - PID
    rightVel = 3.5 + PID
    # print(f"Left Velocity: {leftVel}, Right Velocity: {rightVel}, PID: {PID}, Error: {error}, Delta Error: {delta_err}")
    print(f"{leftVel}, {rightVel}, {PID}, {error}, {delta_err}")
    
    # Set motors velocity
    leftMotor.setVelocity(leftVel)
    rightMotor.setVelocity(rightVel)

    # Show the OpenCV Window
    img_np_resized = cv2.resize(img_np, (int(img_np.shape[1] * scale), int(img_np.shape[0] * scale)))
    cv2.imshow("Gray", gray_resized)
    cv2.imshow("Black Mask", black_mask_resized)
    cv2.imshow("Webots Camera", img_np_resized)

# Close OpenCV Window when the robot restart
cv2.destroyAllWindows()