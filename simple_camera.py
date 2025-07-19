# 导入opencv-python
import cv2
# 导入mediapipe
import mediapipe as mp

# 导入pose solution
mp_pose = mp.solutions.pose

# 导入mediapipe的绘图函数
mp_drawing = mp.solutions.drawing_utils

# 导入模型
pose = mp_pose.Pose(static_image_mode=False,        # 是静态图片还是连续视频帧
                    model_complexity=2,             # 取0,1,2；0最快但性能差，2最慢但性能好
                    smooth_landmarks=True,          # 是否平滑关键点
                    min_detection_confidence=0.5,   # 置信度阈值
                    min_tracking_confidence=0.5)    # 追踪阈值

# 处理单帧的函数
def process_frame(img_BGR):
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    results = pose.process(img_RGB) # 将RGB图片输入模型，获取预测结果
    mp_drawing.draw_landmarks(img_BGR, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) # 可视化
    return img_BGR

# 调用摄像头获取帧
cap = cv2.VideoCapture(0) # Mac电脑的参数为1，Windows电脑的参数为0

cap.open(0)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print('Error')
        break

    frame = process_frame(frame)

    cv2.imshow('camera', frame)

    if cv2.waitKey(1) in [ord('q'),27]: # 按下键盘的 q 或 esc 退出（在英文输入法下）
        break

cap.release()
cv2.destroyAllWindows()