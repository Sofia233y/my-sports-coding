import cv2
import numpy as np
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
# ---------- 计算三点夹角的函数 ----------
def calculate_angle(a, b, c):
    """返回 向量ba 和 向量bc 的夹角，单位：度"""
    a = np.array([a.x, a.y])    # 点a坐标
    b = np.array([b.x, b.y])    # 点b是角的顶点
    c = np.array([c.x, c.y])    # 点c坐标
    ba = a - b
    bc = c - b
    # 余弦定理求角
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine)   # 弧度
    return np.degrees(angle)    # 转成度
# ---------- 主程序 ----------
pose = mp_pose.Pose()
# 如果有摄像头，用 0；如果只有手机拍好的视频，就把 0 换成 "你的视频文件名.mp4"
cap = cv2.VideoCapture("b1.mp4")
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("视频读取完毕或摄像头不可用")
        break
    # 转换颜色格式（BGR → RGB）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        # 画骨架
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))
        # --- 计算并显示右肘关节角度 ---
        right_shoulder = results.pose_landmarks.landmark[12]
        right_elbow    = results.pose_landmarks.landmark[14]
        right_wrist    = results.pose_landmarks.landmark[16]
        angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        # 在图像左上角画角度文字
        cv2.putText(image, f"Elbow angle: {int(angle)} deg",
                    (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)
    # 显示图像
    cv2.imshow('My Pose Demo', image)
    # 按 ESC 键退出
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()