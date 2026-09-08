import cv2
import mediapipe as mp
import math
import pyautogui
import time
import os
import numpy as np

pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False

# Steering Wheel PNG Loader & Generator
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHEEL_PNG_PATH = os.path.join(BASE_DIR, "steering_wheel.png")

def create_default_steering_png(path):
    size = 400
    img = np.zeros((size, size, 4), dtype=np.uint8)
    center = (size // 2, size // 2)

    dark_gray = (35, 35, 35, 255)
    light_gray = (180, 180, 180, 255)
    gold = (0, 215, 255, 255)
    red = (40, 40, 230, 255)

    cv2.circle(img, center, 175, dark_gray, 34)
    cv2.circle(img, center, 192, (15, 15, 15, 255), 2)
    cv2.circle(img, center, 158, (15, 15, 15, 255), 2)
    cv2.ellipse(img, center, (175, 175), 0, -98, -82, red, 32)
    cv2.circle(img, center, 155, gold, 3)
    cv2.circle(img, center, 55, (25, 25, 25, 255), -1)
    cv2.circle(img, center, 55, light_gray, 3)
    cv2.circle(img, center, 25, (10, 10, 10, 255), -1)
    cv2.circle(img, center, 25, gold, 2)

    cv2.rectangle(img, (center[0] - 158, center[1] - 14), (center[0] - 50, center[1] + 14), light_gray, -1)
    cv2.rectangle(img, (center[0] - 158, center[1] - 14), (center[0] - 50, center[1] + 14), (50, 50, 50, 255), 2)

    cv2.rectangle(img, (center[0] + 50, center[1] - 14), (center[0] + 158, center[1] + 14), light_gray, -1)
    cv2.rectangle(img, (center[0] + 50, center[1] - 14), (center[0] + 158, center[1] + 14), (50, 50, 50, 255), 2)

    cv2.rectangle(img, (center[0] - 16, center[1] + 50), (center[0] + 16, center[1] + 158), light_gray, -1)
    cv2.rectangle(img, (center[0] - 16, center[1] + 50), (center[0] + 16, center[1] + 158), (50, 50, 50, 255), 2)

    cv2.circle(img, center, 14, (0, 180, 255, 255), -1)
    cv2.putText(img, "GT", (center[0] - 11, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0, 255), 2)
    cv2.imwrite(path, img)

if not os.path.exists(WHEEL_PNG_PATH):
    create_default_steering_png(WHEEL_PNG_PATH)

steering_wheel_png = cv2.imread(WHEEL_PNG_PATH, cv2.IMREAD_UNCHANGED)

def overlay_rotated_png(bg, img_rgba, center, angle, target_size=(180, 180), opacity=1.0):
    if img_rgba is None:
        return bg
    resized = cv2.resize(img_rgba, target_size, interpolation=cv2.INTER_AREA)
    h_ov, w_ov = resized.shape[:2]
    M = cv2.getRotationMatrix2D((w_ov / 2, h_ov / 2), -angle, 1.0)
    rotated = cv2.warpAffine(resized, M, (w_ov, h_ov), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    cx, cy = center
    x1 = int(cx - w_ov / 2)
    y1 = int(cy - h_ov / 2)
    x2 = x1 + w_ov
    y2 = y1 + h_ov

    bg_h, bg_w = bg.shape[:2]

    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(bg_w, x2)
    crop_y2 = min(bg_h, y2)

    if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
        return bg

    ov_x1 = crop_x1 - x1
    ov_y1 = crop_y1 - y1
    ov_x2 = ov_x1 + (crop_x2 - crop_x1)
    ov_y2 = ov_y1 + (crop_y2 - crop_y1)

    overlay_crop = rotated[ov_y1:ov_y2, ov_x1:ov_x2]
    alpha = (overlay_crop[:, :, 3] / 255.0) * opacity

    for c in range(0, 3):
        bg[crop_y1:crop_y2, crop_x1:crop_x2, c] = (
            alpha * overlay_crop[:, :, c] + (1.0 - alpha) * bg[crop_y1:crop_y2, crop_x1:crop_x2, c]
        )
    return bg

USE_LEGACY_MP = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")

if USE_LEGACY_MP:
    mp_drawing = mp.solutions.drawing_utils
    mphands = mp.solutions.hands

    hands = mphands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55
    )
else:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = os.path.join(BASE_DIR, "hand_landmarker.task")

    if not os.path.exists(model_path):
        import urllib.request
        url = (
            "https://storage.googleapis.com/"
            "mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
        urllib.request.urlretrieve(url, model_path)

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55
    )

    detector = vision.HandLandmarker.create_from_options(options)

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    def draw_hand_landmarks(img, landmarks):
        h, w = img.shape[:2]
        for p1, p2 in HAND_CONNECTIONS:
            pt1 = (int(landmarks[p1].x * w), int(landmarks[p1].y * h))
            pt2 = (int(landmarks[p2].x * w), int(landmarks[p2].y * h))
            cv2.line(img, pt1, pt2, (224, 224, 224), 2)
        for lm in landmarks:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

def init_camera(idx):
    camera = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not camera.isOpened():
        camera.release()
        return None, idx
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(5):
        ret, frame = camera.read()
        time.sleep(0.02)
        if ret and frame is not None:
            return camera, idx
    camera.release()
    return None, idx

def get_working_camera(start_idx=0):
    for idx in range(start_idx, start_idx + 4):
        actual_idx = idx % 4
        cam, i = init_camera(actual_idx)
        if cam is not None:
            print(f"Camera successfully opened at Index {actual_idx}")
            return cam, actual_idx
    cam, i = init_camera(0)
    return cam if cam is not None else cv2.VideoCapture(0), 0

cap, current_cam_idx = get_working_camera(0)

def get_thumb_elevation(lm):
    if not lm:
        return 0.0
    wrist_y = lm[0].y
    middle_mcp_y = lm[9].y
    index_mcp_y = lm[5].y
    thumb_tip_y = lm[4].y
    thumb_mcp_y = lm[2].y

    hand_height = abs(wrist_y - middle_mcp_y)
    if hand_height < 0.05:
        hand_height = 0.2

    elev_index = (index_mcp_y - thumb_tip_y) / hand_height
    elev_mcp = (thumb_mcp_y - thumb_tip_y) / hand_height
    return min(elev_index, elev_mcp)

def thumb_is_up(lm):
    return get_thumb_elevation(lm) > 0.20

pressing_w = False
pressing_a = False
pressing_s = False
pressing_d = False

smooth_angle = 0.0
straight_threshold = 7.0
frames_lost = 0

W_UP_FRAMES = 1
S_UP_FRAMES = 1

right_up_counter = 0
left_up_counter = 0

print("========================================")
print("  GESTURE STEERING HUD (CAMERA CONTROLS)")
print("========================================")
print("PRESS C       = SWITCH CAMERA DEVICE")
print("PRESS 0/1   = SELECT CAMERA INDEX DIRECTLY")
print("BOTH HANDS UP   = ROTATE BOTTOM WHEEL")
print("RIGHT THUMB UP  = W (Gas)")
print("LEFT THUMB UP   = S (Brake)")
print("ESC             = EXIT")

while True:
    ret, image = cap.read()
    if not ret:
        print(f"Camera index {current_cam_idx} lost frame, attempting reconnect...")
        cap.release()
        cap, current_cam_idx = get_working_camera(current_cam_idx + 1)
        continue

    image = cv2.flip(image, 1)
    h, w = image.shape[:2]

    if USE_LEGACY_MP:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        raw_hands = results.multi_hand_landmarks if results.multi_hand_landmarks else []
    else:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detection_result = detector.detect(mp_image)
        raw_hands = detection_result.hand_landmarks if detection_result.hand_landmarks else []

    wheel_pos = (w // 2, h - 110)
    wheel_size = (200, 200)
    wheel_opacity = 0.85

    if raw_hands:
        frames_lost = 0
        sorted_hands = sorted(
            raw_hands,
            key=lambda hd: (hd.landmark[0].x if hasattr(hd, 'landmark') else hd[0].x)
        )

        if len(sorted_hands) >= 2:
            left_hand = sorted_hands[0]
            right_hand = sorted_hands[1]
        else:
            single_hand = sorted_hands[0]
            single_x = single_hand.landmark[0].x if hasattr(single_hand, "landmark") else single_hand[0].x
            if single_x >= 0.5:
                right_hand = single_hand
                left_hand = None
            else:
                left_hand = single_hand
                right_hand = None

        lm_right = (right_hand.landmark if hasattr(right_hand, 'landmark') else right_hand) if right_hand else None
        lm_left = (left_hand.landmark if hasattr(left_hand, 'landmark') else left_hand) if left_hand else None

        if USE_LEGACY_MP:
            if right_hand:
                mp_drawing.draw_landmarks(image, right_hand, mphands.HAND_CONNECTIONS)
            if left_hand:
                mp_drawing.draw_landmarks(image, left_hand, mphands.HAND_CONNECTIONS)
        else:
            if lm_right:
                draw_hand_landmarks(image, lm_right)
            if lm_left:
                draw_hand_landmarks(image, lm_left)

        if lm_right and lm_left:
            x_right, y_right = lm_right[0].x * w, lm_right[0].y * h
            x_left, y_left = lm_left[0].x * w, lm_left[0].y * h

            dx = x_right - x_left
            dy = y_right - y_left
            raw_angle = math.degrees(math.atan2(dy, dx))
            smooth_angle = 0.80 * smooth_angle + 0.20 * raw_angle

            # Thicker glowing steering line connecting hands
            cv2.line(image, (int(x_left), int(y_left)), (int(x_right), int(y_right)), (0, 255, 255), 10)
            cv2.line(image, (int(x_left), int(y_left)), (int(x_right), int(y_right)), (255, 255, 255), 3)

            wheel_pos = (w // 2, h - 110)
            wheel_opacity = 0.95

            if smooth_angle < -straight_threshold:
                if not pressing_a:
                    pyautogui.keyDown("a")
                    pressing_a = True
                if pressing_d:
                    pyautogui.keyUp("d")
                    pressing_d = False
            elif smooth_angle > straight_threshold:
                if not pressing_d:
                    pyautogui.keyDown("d")
                    pressing_d = True
                if pressing_a:
                    pyautogui.keyUp("a")
                    pressing_a = False
            else:
                if pressing_a:
                    pyautogui.keyUp("a")
                    pressing_a = False
                if pressing_d:
                    pyautogui.keyUp("d")
                    pressing_d = False
        else:
            if pressing_a:
                pyautogui.keyUp("a")
                pressing_a = False
            if pressing_d:
                pyautogui.keyUp("d")
                pressing_d = False

        right_up = thumb_is_up(lm_right) if lm_right else False
        left_up = thumb_is_up(lm_left) if lm_left else False

        if right_up:
            right_up_counter += 1
        else:
            right_up_counter = 0

        if left_up:
            left_up_counter += 1
        else:
            left_up_counter = 0

        gas_active = right_up_counter >= W_UP_FRAMES
        brake_active = left_up_counter >= S_UP_FRAMES

        if brake_active:
            gas_active = False

        if gas_active:
            if not pressing_w:
                pyautogui.keyDown("w")
                pressing_w = True
        else:
            if pressing_w:
                pyautogui.keyUp("w")
                pressing_w = False

        if brake_active:
            if not pressing_s:
                pyautogui.keyDown("s")
                pressing_s = True
        else:
            if pressing_s:
                pyautogui.keyUp("s")
                pressing_s = False

        if lm_right:
            rt_x, rt_y = int(lm_right[4].x * w), int(lm_right[4].y * h)
            r_score = get_thumb_elevation(lm_right)
            r_color = (0, 255, 0) if gas_active else (80, 180, 80)
            r_radius = 28 if gas_active else 16
            cv2.circle(image, (rt_x, rt_y), r_radius + 4, (255, 255, 255), 3) # Outline
            cv2.circle(image, (rt_x, rt_y), r_radius, r_color, -1) # Solid Button
            cv2.putText(image, "W (GAS)", (rt_x + 32, rt_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, r_color, 3 if gas_active else 2)

        if lm_left:
            lt_x, lt_y = int(lm_left[4].x * w), int(lm_left[4].y * h)
            l_score = get_thumb_elevation(lm_left)
            l_color = (0, 0, 255) if brake_active else (80, 80, 200)
            l_radius = 28 if brake_active else 16
            cv2.circle(image, (lt_x, lt_y), l_radius + 4, (255, 255, 255), 3) # Outline
            cv2.circle(image, (lt_x, lt_y), l_radius, l_color, -1) # Solid Button
            cv2.putText(image, "S (BRAKE)", (lt_x - 160, lt_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, l_color, 3 if brake_active else 2)

    else:
        frames_lost += 1
        if frames_lost > 5:
            if pressing_w:
                pyautogui.keyUp("w")
                pressing_w = False
            if pressing_a:
                pyautogui.keyUp("a")
                pressing_a = False
            if pressing_s:
                pyautogui.keyUp("s")
                pressing_s = False
            if pressing_d:
                pyautogui.keyUp("d")
                pressing_d = False
            right_up_counter = 0
            left_up_counter = 0

    image = overlay_rotated_png(
        image,
        steering_wheel_png,
        center=wheel_pos,
        angle=smooth_angle,
        target_size=wheel_size,
        opacity=wheel_opacity
    )

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)

    angle_color = (0, 255, 0) if abs(smooth_angle) <= straight_threshold else (0, 165, 255)
    cv2.putText(image, f"STEER: {int(smooth_angle)} deg", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, angle_color, 2)
    cv2.putText(image, f"MOBILE CAM #{current_cam_idx} [Press C to switch]" if current_cam_idx == 0 else f"MAC CAM #{current_cam_idx} [Press C to switch]", (200, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)

    def draw_badge(img, text, pos, active, active_color):
        color = active_color if active else (50, 50, 50)
        cv2.rectangle(img, pos, (pos[0] + 45, pos[1] + 32), color, -1)
        cv2.rectangle(img, pos, (pos[0] + 45, pos[1] + 32), (200, 200, 200), 1)
        cv2.putText(img, text, (pos[0] + 14, pos[1] + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    badge_y = 10
    draw_badge(image, "W", (w - 210, badge_y), pressing_w, (0, 180, 0))
    draw_badge(image, "A", (w - 160, badge_y), pressing_a, (255, 140, 0))
    draw_badge(image, "S", (w - 110, badge_y), pressing_s, (0, 0, 220))
    draw_badge(image, "D", (w - 60, badge_y), pressing_d, (255, 140, 0))

    cv2.imshow("Driving Controls HUD", image)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC
        break
    elif key in [ord("c"), ord("C")]:
        print(f"Switching camera from index {current_cam_idx}...")
        cap.release()
        cap, current_cam_idx = get_working_camera(current_cam_idx + 1)
    elif key in [ord("0"), ord("m"), ord("M")]:
        print("Selecting Mobile Camera Index 0...")
        print("Selecting Camera Index 0...")
        cap.release()
        cap, current_cam_idx = get_working_camera(0)
    elif key == ord("1"):
        print("Selecting Camera Index 1...")
        cap.release()
        cap, current_cam_idx = get_working_camera(1)

for k, state in [("w", pressing_w), ("a", pressing_a), ("s", pressing_s), ("d", pressing_d)]:
    if state:
        pyautogui.keyUp(k)

cap.release()
cv2.destroyAllWindows()
