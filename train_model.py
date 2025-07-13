import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

def load_images_from_folder(folder_path, label=None):
    images = []
    labels = [] if label else None    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (100, 100))    
            images.append(img)
            if label:
                labels.append(label)  
    return (np.array(images), np.array(labels)) if label else np.array(images)

# مسارات بيانات التدريب
benign_folder = r'D:\Dataset_BUSI_with_GT\benign'  
malignant_folder = r'D:\Dataset_BUSI_with_GT\malignant'  
normal_folder = r'D:\Dataset_BUSI_with_GT\normal'  

benign_images, benign_labels = load_images_from_folder(benign_folder, 'benign')
malignant_images, malignant_labels = load_images_from_folder(malignant_folder, 'malignant')
normal_images, normal_labels = load_images_from_folder(normal_folder, 'normal')

images = np.concatenate([benign_images, malignant_images, normal_images], axis=0)
labels = np.concatenate([benign_labels, malignant_labels, normal_labels], axis=0)

images = images.reshape(images.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# حفظ النموذج وال encoder
joblib.dump(clf, 'random_forest_model.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

# اختبار الدقة
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
