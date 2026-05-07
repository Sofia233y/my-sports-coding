import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
# 如果你有摄像头，用 0；如果想测试视频文件，把 0 换成 "你的视频文件名.mp4"
cap = cv2.VideoCapture("b1.mp4")
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("视频读取完毕或摄像头不可用")
        break
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('My Pose Demo', image)
    # 按 ESC 键退出
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()