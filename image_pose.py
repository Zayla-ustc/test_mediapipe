import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe Pose 和绘图工具
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- 配置 ---
# 输入图片路径
IMAGE_PATH = "E:\\0718-2249\\0718-225149_Nakh\\tmp_result.png" # 替换成你的图片路径

# MediaPipe Pose 模型配置
# static_image_mode=True 表示输入是静态图片
# min_detection_confidence 表示最低检测置信度
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# --- 主程序 ---

# 1. 读取图片
try:
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"错误：无法读取图片，请检查路径是否正确: {IMAGE_PATH}")
except FileNotFoundError as e:
    print(e)
    exit()

# 获取图片尺寸
image_height, image_width, _ = image.shape

# 2. 运行姿态检测
# MediaPipe 需要 RGB 格式的图片，而 OpenCV 读取的是 BGR 格式，所以需要转换
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)

# 3. 绘制结果
# 如果检测到了姿态
if results.pose_landmarks:
    print("成功检测到人体姿态！")
    
    # 在原始图片上绘制姿态关键点和连接线
    # 我们创建一个副本进行绘制，以保留原始图片
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        # 自定义关键点样式
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        # 自定义连接线样式
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )
    
    # 4. 显示图片
    # 使用 cv2.imshow 在窗口中显示处理后的图片
    cv2.imshow('MediaPipe Pose Detection', annotated_image)
    
    # 等待用户按键后关闭窗口 (按任意键退出)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    
    # 保存处理后的图片（可选）
    output_path = 'output_image.jpg'
    cv2.imwrite(output_path, annotated_image)
    print(f"结果已保存到: {output_path}")

else:
    print(f"在图片 '{IMAGE_PATH}' 中未能检测到人体姿态。")

# 释放资源
pose.close()
cv2.destroyAllWindows()