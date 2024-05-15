from controller import Robot, Camera, Motor
import cv2
import numpy as np

# Inisialisasi Variabel
MAX_SPEED = 6.28

# Inisialisasi Robot.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# States
states = ['forward', 'turn_right', 'turn_left']
current_state = states[0]

# Fungsi Camera getDevice
camera = robot.getDevice('camera')  # Inisialisasi kamera
camera.enable(timestep)  # Aktifkan kamera dengan timestep yang ditentukan

# Motors   
leftMotor = robot.getDevice('left wheel motor')  # Inisialisasi motor roda kiri
rightMotor = robot.getDevice('right wheel motor')  # Inisialisasi motor roda kanan
leftMotor.setPosition(float('inf'))  # Set posisi motor kiri ke tak hingga
rightMotor.setPosition(float('inf'))  # Set posisi motor kanan ke tak hingga
leftMotor.setVelocity(0.0)  # Atur kecepatan awal motor kiri menjadi 0.0
rightMotor.setVelocity(0.0)  # Atur kecepatan awal motor kanan menjadi 0.0

while robot.step(timestep) != -1:  # Loop utama, menjalankan simulasi sampai selesai atau dihentikan
    # Memproses data
    image = camera.getImage()  # Ambil gambar dari kamera
    img_np = np.frombuffer(image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))  # Konversi gambar ke array NumPy
    img_np = img_np[:, :, :3]  # Ambil 3 saluran warna (R, G, B) dari gambar RGBA
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # Konversi warna BGR ke RGB

    # Mengonversi citra ke skala abu-abu
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]

    # Filter warna-warna non-hitam pada gambar asli
    _, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_np_resize1 = cv2.resize(black_mask, (int(img_np.shape[1] * 0.4), int(img_np.shape[0] * 0.4)))
    cv2.imshow("black_mask",img_np_resize1)
    # Temukan kontur pada gambar hasil thresholding
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inisialisasi state
    current_state = 'forward'
    
    # Jika terdeteksi kontur, hitung jarak terdekat dari kontur ke tengah citra
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)  # Pilih kontur terbesar
        moments = cv2.moments(contour)  # Hitung momen kontur
        ypot = int(height/2)
        xpot = int(width/2)
        cv2.circle(img_np, (xpot, ypot), 5, (255, 0, 0), -1)
        if moments['m00'] != 0:
            x = int(moments['m10'] / moments['m00'])  # Hitung posisi tengah kontur
            y = int(height/2)#int(moments['m01'] / moments['m00'])
            cv2.circle(img_np, (x, y), 5, (0, 255, 0), -1)  # Gambarkan posisi tengah kontur dengan warna hijau

            # Hitung jarak terdekat dari posisi tengah kontur ke tengah citra
            center_x = img_np.shape[1] // 2

            # Jika posisi kontur lebih besar dari 1/3 lebar citra, ganti state menjadi 'turn_right' atau 'turn_left'
            if x < center_x - center_x // 3:
                current_state = 'turn_left'
            elif x > center_x + center_x // 3:
                current_state = 'turn_right'

    # Atur kecepatan motor sesuai dengan nilai kecepatan yang telah ditentukan
    if current_state == 'forward':
        leftSpeed = MAX_SPEED
        rightSpeed = MAX_SPEED
    elif current_state == 'turn_right':
        leftSpeed = MAX_SPEED / 2
        rightSpeed = MAX_SPEED
    elif current_state == 'turn_left':
        leftSpeed = MAX_SPEED
        rightSpeed = MAX_SPEED / 2

    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)

    # Tampilkan gambar di jendela "Webots Camera"
    img_np_resize = cv2.resize(img_np, (int(img_np.shape[1] * 0.4), int(img_np.shape[0] * 0.4)))
    cv2.imshow("Webots Camera", img_np_resize)

    # Jika tombol 'q' ditekan, hentikan loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(y)
    print(f"M.Ki : {leftSpeed:.2f} || M.Ka : {rightSpeed:.2f}")
# Tutup semua jendela yang terbuka ketika loop selesai
cv2.destroyAllWindows()