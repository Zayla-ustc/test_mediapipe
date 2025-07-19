# 导入opencv-python
import cv2
# 导入mediapipe
import mediapipe as mp
import time

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
    start_time = time.time()
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    results = pose.process(img_RGB) # 将RGB图片输入模型，获取预测结果
    ## 获取图片长宽
    h, w = img_BGR.shape[0], img_BGR.shape[1]

    ## 遍历33个关键点
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img_BGR, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for i in range(33):
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            cz = results.pose_landmarks.landmark[i].z

            radius = 5
            if i == 0:  # 鼻尖
                img_BGR = cv2.circle(img_BGR, (cx, cy), radius, (0, 0, 225), -1)    # -1表示填充
            elif i in [11, 12]:  # 肩膀
                img_BGR = cv2.circle(img_BGR, (cx, cy), radius, (223, 155, 6), -1)
            elif i in [23, 24]:  # 髋关节
                img_BGR = cv2.circle(img_BGR, (cx, cy), radius, (1, 240, 255), -1)
            elif i in [13, 14]:  # 胳膊肘
                img_BGR = cv2.circle(img_BGR, (cx, cy), radius, (140, 47, 240), -1)
            elif i in [25, 26]:  # 膝盖
                img_BGR = cv2.circle(img_BGR, (cx, cy), radius, (0, 0, 225), -1)
            elif i in [15, 16, 27, 28]:  # 手腕和脚腕
                img_BGR = cv2.circle(img_BGR, (cx, cy), radius, (223, 155, 60), -1)
            elif i in [17, 19, 21]:  # 左手
                img_BGR = cv2.circle(img_BGR, (cx, cy), radius, (94, 218, 121), -1)
            elif i in [18, 20, 22]:  # 右手
                img_BGR = cv2.circle(img_BGR, (cx, cy), radius, (16, 144, 247), -1)
            elif i in [27, 29, 31]:  # 左脚
                img_BGR = cv2.circle(img_BGR, (cx, cy), radius, (29, 123, 243), -1)
            elif i in [28, 30, 32]:  # 右脚
                img_BGR = cv2.circle(img_BGR, (cx, cy), radius, (193, 182, 255), -1)
            elif i in [9, 10]:  # 嘴
                img_BGR = cv2.circle(img_BGR, (cx, cy), radius, (205, 235, 255), -1)
            elif i in [1, 2, 3, 4, 5, 6, 7, 8]:  # 眼和脸颊
                img_BGR = cv2.circle(img_BGR, (cx, cy), radius, (94, 218, 121), -1)
            else:   # 其它关键点
                img_BGR = cv2.circle(img_BGR, (cx, cy), radius, (0, 225, 0), -1)
    else:
        scaler = 1
        failure_str = 'No Person'
        img_BGR = cv2.putText(img_BGR, failure_str, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 0))
    end_time = time.time()
    FPS = 1 / (end_time - start_time)
    scaler = 1
    # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，文字大小，颜色，文字粗细
    img_BGR = cv2.putText(img_BGR, 'FPS ' + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 0))
    return img_BGR

# 调用摄像头获取帧
cap = cv2.VideoCapture(1) # Mac电脑的参数为1，Windows电脑的参数为0

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