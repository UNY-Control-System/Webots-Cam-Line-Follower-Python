from controller import Robot, Camera, Motor
import cv2
import numpy as np
import skfuzzy as fuzz


# Inisialisasi Variabel
MAX_SPEED = 2
e_prev = 0

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
    time = robot.getTime()
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

    # Temukan kontur pada gambar hasil thresholding
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inisialisasi state
    current_state = 'forward'
    
    # Jika terdeteksi kontur, hitung jarak terdekat dari kontur ke tengah citra
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)  # Pilih kontur terbesar
        moments = cv2.moments(contour)  # Hitung momen kontur
        if moments['m00'] != 0:
            x = int(moments['m10'] / moments['m00'])  # Hitung posisi tengah kontur
            y = int(moments['m01'] / moments['m00'])
            ijo = cv2.circle(img_np, (x, y), 5, (0, 255, 0), -1)  # Gambarkan posisi tengah kontur dengan warna hijau
            
            x_pot = int(width/2)
            y_pot = int(height*0.75)
            sp = cv2.circle(img_np, (x_pot, y), 10, (0, 0, 255), -1)
            # Hitung jarak terdekat dari posisi tengah kontur ke tengah citra
            center_x = img_np.shape[1] // 2
            
            error = x-x_pot
            delta_error = e_prev-error
            
            if time != 0.032 :
            
                # Define the range for input and output
                input_range = np.arange(-640, 640, 0.1)
                input_range1 = np.arange(-640, 640, 0.1)
                output_range = np.arange(-7, 7, 0.01)
                
                # Define fuzzy membership functions for input error
                input_kiri = fuzz.trapmf(input_range, [-640, -640, -300, -200])
                input_agak_kiri = fuzz.trimf(input_range, [-320, -180, -40])
                input_tengah = fuzz.trapmf(input_range, [-60, -40, 40, 60])
                input_agak_kanan = fuzz.trimf(input_range, [15, 115, 215])
                input_kanan = fuzz.trapmf(input_range, [200, 320, 640, 640])
                
                # Define fuzzy membership functions for input delta error
                input_kiri1 = fuzz.trapmf(input_range1, [-640, -640, -300, -200])
                input_agak_kiri1 = fuzz.trimf(input_range1, [-320, -180, -40])
                input_tengah1 = fuzz.trapmf(input_range1,[-60, -40, 40, 60])
                input_agak_kanan1 = fuzz.trimf(input_range1,[40, 180, 320])
                input_kanan1 = fuzz.trapmf(input_range1, [200, 320, 640, 640])
                
                # Define fuzzy membership functions for output
                output_sangatkiri = fuzz.trapmf(output_range, [-7, -7, -6, -5])
                output_kiri = fuzz.trapmf(output_range, [-6, -5, -4, -3])
                output_agakkiri = fuzz.trapmf(output_range,  [-4, -3, -2, -1])
                output_tengah = fuzz.trapmf(output_range, [-2, -1, 1, 2])
                output_agakkanan = fuzz.trapmf(output_range, [1, 2, 3, 4])
                output_kanan = fuzz.trapmf(output_range, [3, 4, 5, 6])
                output_sangatkanan = fuzz.trapmf(output_range, [5, 6, 7, 7])
                
                errorkiri = fuzz.interp_membership(input_range, input_kiri , error)
                erroragakkiri = fuzz.interp_membership(input_range, input_agak_kiri , error)
                errortengah = fuzz.interp_membership(input_range, input_tengah , error)
                erroragakkanan = fuzz.interp_membership(input_range, input_agak_kanan , error)
                errorkanan = fuzz.interp_membership(input_range, input_kanan , error)
                
                delerror_kiri =fuzz.interp_membership(input_range1, input_kiri1 , delta_error)
                delerror_agakkiri =fuzz.interp_membership(input_range1, input_agak_kiri1 , delta_error)
                delerror_tengah =fuzz.interp_membership(input_range1, input_tengah1 , delta_error)
                delerror_agakkanan =fuzz.interp_membership(input_range1, input_agak_kanan1 , delta_error)
                delerror_kanan =fuzz.interp_membership(input_range1, input_kanan1 , delta_error)
                
                rule1 = np.fmin(errorkiri, delerror_kiri)
                rule2 = np.fmin(errorkiri, delerror_agakkiri)
                rule3 = np.fmin(errorkiri, delerror_tengah)
                rule4 = np.fmin(erroragakkiri, delerror_kiri)
                rule5 = np.fmin(erroragakkiri, delerror_agakkiri)
                rule6 = np.fmin(erroragakkiri, delerror_tengah)
                rule7 = np.fmin(erroragakkiri, delerror_agakkanan)
                rule8 = np.fmin(errortengah, delerror_kiri)
                rule9 = np.fmin(errortengah, delerror_agakkiri)
                rule10 = np.fmin(errortengah, delerror_tengah)
                rule11 = np.fmin(errortengah, delerror_agakkanan)
                rule12 = np.fmin(errortengah, delerror_kanan)
                rule13 = np.fmin(erroragakkanan, delerror_agakkiri)
                rule14 = np.fmin(erroragakkanan, delerror_tengah)
                rule15 = np.fmin(erroragakkanan, delerror_agakkanan)
                rule16 = np.fmin(erroragakkanan, delerror_kanan)
                rule17 = np.fmin(errorkanan, delerror_tengah)
                rule18 = np.fmin(errorkanan, delerror_agakkanan)
                rule19 = np.fmin(errorkanan, delerror_kanan)
                
                
                active_rule1 = np.fmin(rule1, output_sangatkiri)
                active_rule2 = np.fmin(np.fmax(rule2, np.fmax(rule4, rule8)), output_kiri)
                active_rule3 = np.fmin(np.fmax(rule3, np.fmax(rule5, np.fmax(rule9, rule13))), output_agakkiri)
                active_rule4 = np.fmin(np.fmax(rule6, np.fmax(rule10, rule14)), output_tengah)
                active_rule5 = np.fmin(np.fmax(rule7, np.fmax(rule11, np.fmax(rule15, rule17))), output_agakkanan)
                active_rule6 = np.fmin(np.fmax(rule12, np.fmax(rule16, rule18)), output_kanan)
                active_rule7 = np.fmin(rule19, output_sangatkanan)
                
                
                
                aggregated = np.fmax(active_rule1, np.fmax(active_rule2, np.fmax(active_rule3, np.fmax(active_rule4, np.fmax(active_rule5, np.fmax(active_rule6, active_rule7))))))
                
                # Defuzzify
                out = fuzz.defuzz(output_range, aggregated, 'centroid')
                out_activation = fuzz.interp_membership(output_range, aggregated, out)
                leftVel = MAX_SPEED - out
                rightVel = MAX_SPEED + out
            
                leftMotor.setVelocity(leftVel)
                rightMotor.setVelocity(rightVel)

    # Tampilkan gambar di jendela "Webots Camera"
    img_np_resize = cv2.resize(img_np, (int(img_np.shape[1] * 0.4), int(img_np.shape[0] * 0.4)))
    cv2.imshow("Webots Camera", img_np_resize)
    
    print(error, delta_error)
    # Jika tombol 'q' ditekan, hentikan loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup semua jendela yang terbuka ketika loop selesai
cv2.destroyAllWindows()