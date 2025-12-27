import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import threading

# Initialize Mediapipe Face Mesh Model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Get Screen Size
screen_w, screen_h = pyautogui.size()

# Cursor Position Variables (Smoother Movement)
prev_x, prev_y = pyautogui.position()
smooth_factor = 0.2  # Reduce sudden jumps
precision_mode = False  # Toggle for fine control

# Click & Drag Variables
cooldown_time = 0.5
last_click_time = 0
dragging = False  # Track dragging state

# Blink Detection Variables
blink_threshold = 6
blink_frames = 2
left_blink_counter = 0
right_blink_counter = 0

# Video Capture
video_capture = cv2.VideoCapture(0)

# Function to Move Mouse in a Thread
def move_mouse(x, y, duration):
    threading.Thread(target=pyautogui.moveTo, args=(x, y, duration)).start()

# Function to Perform Right Click in a Thread
def right_click():
    threading.Thread(target=lambda: (pyautogui.mouseDown(button='right'), pyautogui.mouseUp(button='right'))).start()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror Effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    frame_h, frame_w, _ = frame.shape
    current_time = time.time()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # **Use Nose Tip for Cursor Movement**
            nose_tip = face_landmarks.landmark[1]
            x = int(nose_tip.x * frame_w)
            y = int(nose_tip.y * frame_h)

            # **Fix Cursor Mapping for Precision**
            screen_x = np.interp(nose_tip.x, [0.3, 0.7], [0, screen_w])  # Reduced range for precise control
            screen_y = np.interp(nose_tip.y, [0.3, 0.7], [0, screen_h])

            # **Fine-Tuning Movements**
            prev_x = prev_x * (1 - smooth_factor) + screen_x * smooth_factor
            prev_y = prev_y * (1 - smooth_factor) + screen_y * smooth_factor

            # **Move Mouse Using a Separate Thread**
            move_mouse(prev_x, prev_y, duration=0.05 if not precision_mode else 0.1)

            # **Eye Blink Detection**
            left_top = int(face_landmarks.landmark[159].y * frame_h)
            left_bottom = int(face_landmarks.landmark[145].y * frame_h)
            right_top = int(face_landmarks.landmark[386].y * frame_h)
            right_bottom = int(face_landmarks.landmark[374].y * frame_h)

            left_eye_opening = abs(left_top - left_bottom)
            right_eye_opening = abs(right_top - right_bottom)

            # **Left Blink for Dragging**
            if left_eye_opening < blink_threshold:
                left_blink_counter += 1
                if left_blink_counter >= blink_frames and (current_time - last_click_time) > cooldown_time:
                    if not dragging:
                        pyautogui.mouseDown(button='left')
                        dragging = True
                        print("Left Blink Detected: Dragging Started")
                    last_click_time = current_time
            else:
                if dragging:
                    pyautogui.mouseUp(button='left')
                    dragging = False
                    print("Left Blink Released: Dragging Stopped")
                left_blink_counter = 0

            # **Right Blink for Right Click**
            if right_eye_opening < blink_threshold:
                right_blink_counter += 1
                if right_blink_counter >= blink_frames and (current_time - last_click_time) > cooldown_time:
                    right_click()
                    print("Right Blink Detected: Right Click Performed")
                    last_click_time = current_time
                    right_blink_counter = 0
            else:
                right_blink_counter = 0

    # Display Output
    cv2.imshow("Precise Eye Mouse Control", frame)

    # **Exit when 'Esc' is pressed**
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == ord('p'):  # 'P' key to toggle precision mode
        precision_mode = not precision_mode
        print("Precision Mode:", "ON" if precision_mode else "OFF")

video_capture.release()
cv2.destroyAllWindows()
