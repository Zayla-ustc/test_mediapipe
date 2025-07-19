
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

# ----------------- 特征提取函数 (核心) -----------------
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

    # --- 1. 姿态归一化 (平移和缩放不变性) ---
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

# ----------------- 主训练流程 -----------------
def main_train():
    DATASET_PATH = 'dataset'
    # 标签及其对应的数字ID
    LABELS = {'standing': 0, 'sitting': 1, 'lying': 2}
    MODEL_SAVE_PATH = 'pose_classifier_xgb.joblib'

    features_list = []
    labels_list = []

    pose_detector = mp.solutions.pose.Pose(static_image_mode=True)
    print("🚀 开始处理图片并提取特征...")

    for label_name, label_id in LABELS.items():
        folder_path = os.path.join(DATASET_PATH, label_name)
        if not os.path.exists(folder_path):
            print(f"警告: 找不到文件夹 {folder_path}，跳过。")
            continue

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None: continue

            results = pose_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                try:
                    features = extract_features_advanced(results.pose_landmarks.landmark)
                    if features is not None:
                        features_list.append(features)
                        labels_list.append(label_id)
                except Exception as e:
                    print(f"处理文件 {filename} 时出错并跳过: {e}")

    pose_detector.close()
    print("✅ 特征提取完成！")
    
    if not features_list:
        print("错误: 未能从数据集中提取任何特征。请检查图片路径和内容。")
        return

    # 创建数据集
    X = pd.DataFrame(features_list)
    y = pd.Series(labels_list, name="label")

    # 划分训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\n🤖 开始训练 XGBoost 模型...")
    print(f"总样本数: {len(X)}, 训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")

    # 训练模型
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # 评估模型
    print(" 正在评估模型...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎉 模型在测试集上的准确率: {accuracy:.4f}")

    # 保存模型
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\n 模型已成功保存到: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main_train()