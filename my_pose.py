import cv2
import numpy as np
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# ---------- 三点夹角计算 ----------
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine)
    return np.degrees(angle)

# ---------- 主程序 ----------
pose = mp_pose.Pose()

# 0 用摄像头，或者填 "你的视频文件名.mp4"
cap = cv2.VideoCapture("b1.mp4")

# 创建数据记录文件
csv_file = open('shot_analysis.csv', 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['time', 'right_elbow', 'right_knee', 'right_hip',
                 'left_elbow', 'left_knee', 'left_hip'])

# ---------- 暂停状态控制变量 ----------
is_paused = False          # 当前是否暂停
# 用于在暂停时暂存当前帧的画面，避免窗口空白
paused_frame = None

print("按【空格键】暂停/继续，按【ESC】退出")

while cap.isOpened():
    # ---- 如果不是暂停状态，就读取下一帧 ----
    if not is_paused:
        success, image = cap.read()
        if not success:
            print("视频结束或摄像头无法读取")
            break
    else:
        # 暂停的时候，反复等待用户按键
        key = cv2.waitKey(0) & 0xFF
        if key == 32:               # 空格键
            is_paused = False       # 恢复播放
            # 继续读取下一帧
            success, image = cap.read()
            if not success:
                break
        elif key == 27:             # ESC 直接退出
            break
        else:
            # 不是空格也不是ESC，就继续暂停着
            continue

    # ---- 图像处理（骨架检测 + 角度计算） ----
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # 画骨架
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2))

        # 计算 6 个关键角度
        right_elbow = calculate_angle(lm[12], lm[14], lm[16])
        right_knee  = calculate_angle(lm[24], lm[26], lm[28])
        right_hip   = calculate_angle(lm[12], lm[24], lm[26])
        left_elbow  = calculate_angle(lm[11], lm[13], lm[15])
        left_knee   = calculate_angle(lm[23], lm[25], lm[27])
        left_hip    = calculate_angle(lm[11], lm[23], lm[25])

        # 在图像上显示角度
        y_base = 30
        dy = 35
        cv2.putText(image, f"R_Elbow: {right_elbow:.1f} deg",
                    (15, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(image, f"R_Knee : {right_knee:.1f} deg",
                    (15, y_base+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(image, f"R_Hip  : {right_hip:.1f} deg",
                    (15, y_base+2*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(image, f"L_Elbow: {left_elbow:.1f} deg",
                    (15, y_base+3*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(image, f"L_Knee : {left_knee:.1f} deg",
                    (15, y_base+4*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(image, f"L_Hip  : {left_hip:.1f} deg",
                    (15, y_base+5*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # 数据写入 CSV
        if not is_paused:   # 只在正常播放时记录数据，避免重复写入同一帧的数据
            now = time.time()
            writer.writerow([now, right_elbow, right_knee, right_hip,
                             left_elbow, left_knee, left_hip])

    # ---- 如果是暂停状态，画上提示文字 ----
    if is_paused:
        cv2.putText(image, "PAUSED", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
        cv2.putText(image, "Press SPACE to continue", (50, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # ---- 显示图像 ----
    cv2.imshow('Full Body Sports Analysis', image)

    # ---- 按键控制 ----
    key = cv2.waitKey(5) & 0xFF
    if key == 27:          # ESC 退出
        break
    elif key == 32:        # 空格键切换暂停/继续
        is_paused = not is_paused
        if is_paused:
            # 刚进入暂停状态时，保存当前帧副本，防止窗口消失
            paused_frame = image.copy()

# 清理
csv_file.close()
cap.release()
cv2.destroyAllWindows()
print("分析完成，数据已保存到 shot_analysis.csv")
import matplotlib.pyplot as plt
try:
    df = pd.read_csv('shot_analysis.csv')   # 需要 pip install pandas
    plt.figure(figsize=(10, 5))
    plt.plot(df['time'] - df['time'][0], df['right_elbow'], label='Right Elbow')
    plt.plot(df['time'] - df['time'][0], df['right_knee'], label='Right Knee')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.legend()
    plt.title('Shooting Motion Analysis')
    plt.savefig('motion_report.png')
    print("运动报告已生成：motion_report.png")
except Exception as e:
    print("图表生成失败，请确保安装了 pandas 和 matplotlib")