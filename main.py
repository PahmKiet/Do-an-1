# import cv2
# import mediapipe as mp
#
# # 1. Khoi tao bo detect tay
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.5
# )
# mp_draw = mp.solutions.drawing_utils
#
# # 2. Mo Camera
# cap = cv2.VideoCapture(0)
#
# print("HE THONG DA SAN SANG! Dua tay len nao ...")
#
# while cap.isOpened():
#     success, img = cap.read()
#     if not success: break
#
#     # Lat anh cho giong soi guong
#     img = cv2.flip(img, 1)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     # AI xu ly hinh anh
#     results = hands.process(img_rgb)
#
#     # 3. Ve khung xuong neu thay tay
#     if results.multi_hand_landmarks:
#         for hand_lms in results.multi_hand_landmarks:
#             # Ve 21 diem va cac duong noi mau xanh
#             mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
#             print(">>> DA THAY TAY!           ", end="\r")
#
#     # Hien thi len man hinh
#     cv2.imshow("DO AN PARKINSON - BAO DUONG HUST", img)
#
#     # Nhan 'q' de thoat
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
# print("\nDA DONG HE THONG.")
import cv2
import mediapipe as mp
import math

# Khoi tao MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1)
    h, w, c = img.shape # Lay kich thuoc man hinh
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # 1. Lay toa do ngon tro (ID 8) va ngon cai (ID 4)
            index_finger = hand_lms.landmark[8]
            thumb_finger = hand_lms.landmark[4]

            # Chuyen toa do tu ti le (0-1) sang Pixel
            cx8, cy8 = int(index_finger.x * w), int(index_finger.y * h)
            cx4, cy4 = int(thumb_finger.x * w), int(thumb_finger.y * h)

            # 2. Tinh khoang cach giua 2 dau ngon tay (Dung cong thuc Euclid)
            distance = math.sqrt((cx8 - cx4)**2 + (cy8 - cy4)**2)

            # 3. Ve hieu ung
            cv2.circle(img, (cx8, cy8), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx4, cy4), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (cx8, cy8), (cx4, cy4), (0, 255, 0), 3)

            # Hien thi khoang cach len man hinh
            cv2.putText(img, f"Khoang cach: {int(distance)}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("DO AN PARKINSON - TRICH XUAT DU LIEU", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()