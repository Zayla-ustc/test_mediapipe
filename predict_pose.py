import cv2
import mediapipe as mp
import numpy as np
import joblib

# ----------------- 特征提取函数 (必须与训练时完全一致) -----------------
# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """计算三点之间的角度 (b为顶点)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return angle if angle <= 180 else 360-angle

def extract_features_advanced(landmarks):
    """从MediaPipe的关键点中提取一个高级特征向量。"""
    if not landmarks:
        return None

    # --- 1. 姿态归一化 ---
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    hip_center_x = (left_hip.x + right_hip.x) / 2
    hip_center_y = (left_hip.y + right_hip.y) / 2

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
    
    torso_size = np.sqrt((shoulder_center_x - hip_center_x)**2 + (shoulder_center_y - hip_center_y)**2)
    if torso_size < 1e-6: torso_size = 1

    normalized_coords = []
    for landmark in landmarks:
        norm_x = (landmark.x - hip_center_x) / torso_size
        norm_y = (landmark.y - hip_center_y) / torso_size
        normalized_coords.extend([norm_x, norm_y])

    # --- 2. 提取双边对称的关键角度 ---
    points = {name: landmarks[getattr(mp_pose.PoseLandmark, name).value] for name in [
        'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 
        'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE'
    ]}
    
    angle_features = [
        calculate_angle([points['LEFT_SHOULDER'].x, points['LEFT_SHOULDER'].y], [points['LEFT_ELBOW'].x, points['LEFT_ELBOW'].y], [points['LEFT_WRIST'].x, points['LEFT_WRIST'].y]),
        calculate_angle([points['RIGHT_SHOULDER'].x, points['RIGHT_SHOULDER'].y], [points['RIGHT_ELBOW'].x, points['RIGHT_ELBOW'].y], [points['RIGHT_WRIST'].x, points['RIGHT_WRIST'].y]),
        calculate_angle([points['LEFT_ELBOW'].x, points['LEFT_ELBOW'].y], [points['LEFT_SHOULDER'].x, points['LEFT_SHOULDER'].y], [points['LEFT_HIP'].x, points['LEFT_HIP'].y]),
        calculate_angle([points['RIGHT_ELBOW'].x, points['RIGHT_ELBOW'].y], [points['RIGHT_SHOULDER'].x, points['RIGHT_SHOULDER'].y], [points['RIGHT_HIP'].x, points['RIGHT_HIP'].y]),
        calculate_angle([points['LEFT_SHOULDER'].x, points['LEFT_SHOULDER'].y], [points['LEFT_HIP'].x, points['LEFT_HIP'].y], [points['LEFT_KNEE'].x, points['LEFT_KNEE'].y]),
        calculate_angle([points['RIGHT_SHOULDER'].x, points['RIGHT_SHOULDER'].y], [points['RIGHT_HIP'].x, points['RIGHT_HIP'].y], [points['RIGHT_KNEE'].x, points['RIGHT_KNEE'].y]),
        calculate_angle([points['LEFT_HIP'].x, points['LEFT_HIP'].y], [points['LEFT_KNEE'].x, points['LEFT_KNEE'].y], [points['LEFT_ANKLE'].x, points['LEFT_ANKLE'].y]),
        calculate_angle([points['RIGHT_HIP'].x, points['RIGHT_HIP'].y], [points['RIGHT_KNEE'].x, points['RIGHT_KNEE'].y], [points['RIGHT_ANKLE'].x, points['RIGHT_ANKLE'].y])
    ]

    # --- 3. 提取全局形状特征 ---
    all_x = [lm.x for lm in landmarks]; all_y = [lm.y for lm in landmarks]
    pose_width = max(all_x) - min(all_x); pose_height = max(all_y) - min(all_y)
    aspect_ratio = pose_width / pose_height if pose_height > 0 else 0
    shape_features = [aspect_ratio]

    # --- 4. 组合所有特征 ---
    return np.concatenate([normalized_coords, angle_features, shape_features]).flatten()


# ----------------- 主预测流程 -----------------
def main_predict():
    MODEL_PATH = 'pose_classifier_xgb.joblib'
    IMAGE_TO_PREDICT = 'test_image.jpg' # 替换成你要预测的图片
    # 标签ID需要和训练时保持一致
    LABEL_NAMES = {0: 'Standing', 1: 'Sitting', 2: 'Lying'}

    # 加载模型
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 '{MODEL_PATH}'。请先运行 `train_model.py` 来生成模型。")
        return

    # 初始化MediaPipe
    mp_drawing = mp.solutions.drawing_utils
    pose_detector = mp.solutions.pose.Pose(min_detection_confidence=0.5)

    # 读取并处理图片
    image = cv2.imread(IMAGE_TO_PREDICT)
    if image is None:
        print(f"错误: 无法读取图片 '{IMAGE_TO_PREDICT}'")
        return

    results = pose_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    status = "No Person Detected"

    if results.pose_landmarks:
        try:
            features = extract_features_advanced(results.pose_landmarks.landmark)
            if features is not None:
                # 使用模型预测
                prediction_id = model.predict(features.reshape(1, -1))[0]
                status = LABEL_NAMES.get(prediction_id, "Unknown Pose")
            else:
                status = "Feature Extraction Failed"
        except Exception as e:
            status = "Prediction Error"
            print(f"预测时出错: {e}")
        
        # 绘制骨架
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    
    # 在图片上显示结果
    cv2.putText(image, f"Status: {status}", (10, 50), 
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, cv2.LINE_AA)

    print(f"预测结果: {status}")
    cv2.imshow('Pose Classification Prediction', image)
    cv2.waitKey(0)

    pose_detector.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main_predict()