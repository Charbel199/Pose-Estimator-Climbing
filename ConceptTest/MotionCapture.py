import cv2
import mediapipe as mp
from typing import List
import time


def draw_rectangle_around_points(img,
                                 points: List,
                                 color=(0, 255, 0),
                                 offset=0):
    x_min = min(p[0] for p in points) - offset
    y_min = min(p[1] for p in points) - offset
    x_max = max(p[0] for p in points) + offset
    y_max = max(p[1] for p in points) + offset
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)


mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()

LEFT_HAND = [15, 17, 19, 21]
RIGHT_HAND = [16, 18, 20, 22]
LEFT_FOOT = [27, 29, 31]
RIGHT_FOOT = [28, 30, 32]
pTime = 0

RESOURCES_DIR = "./resources"


cap = cv2.VideoCapture(f"{RESOURCES_DIR}/ClimbingVideo.mp4")
while True:
    success, img = cap.read()
    if img is None:
        print("Done...")
        cv2.destroyAllWindows()
        break

    static_img = None
    static_img = cv2.imread(f"{RESOURCES_DIR}/DropKnee4.png")
    # static_img = cv2.resize(static_img, (1200, 800))
    if static_img is not None:
        img = static_img

    h, w, c = img.shape

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        # Draw landmarks
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        left_hand_points = []
        right_hand_points = []
        left_foot_points = []
        right_foot_points = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            point = (int(lm.x * w), int(lm.y * h))
            if id in LEFT_HAND:
                left_hand_points.append(point)
            if id in RIGHT_HAND:
                right_hand_points.append(point)
            if id in LEFT_FOOT:
                left_foot_points.append(point)
            if id in RIGHT_FOOT:
                right_foot_points.append(point)

        # Draw Green squares around hands
        color = (0, 255, 0)
        offset = 15
        draw_rectangle_around_points(img,
                                     left_hand_points,
                                     color=color,
                                     offset=offset)
        draw_rectangle_around_points(img,
                                     right_hand_points,
                                     color=color,
                                     offset=offset)
        # Draw Purple squares around feet
        color = (255, 0, 255)
        draw_rectangle_around_points(img,
                                     left_foot_points,
                                     color=color,
                                     offset=offset)
        draw_rectangle_around_points(img,
                                     right_foot_points,
                                     color=color,
                                     offset=offset)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    cv2.imshow("ClimbingPerson", img)

    if cv2.waitKey(0 if static_img is not None else 1) == ord('q'):
        cv2.destroyAllWindows()
        break
