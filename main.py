import os
import cv2
import numpy as np
import time
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from concurrent.futures import ThreadPoolExecutor
import shutil

input_folder = './two_players_top'
output_folder_player1 = './oujll'
output_folder_player2 = './oudll'

os.makedirs(output_folder_player1, exist_ok=True)
os.makedirs(output_folder_player2, exist_ok=True)

total_start_time = time.time()

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)  
model = Model(inputs=base_model.input, outputs=x)

def extract_deep_features(image):
    image_resized = cv2.resize(image, (224, 224))
    image_preprocessed = preprocess_input(image_resized)
    image_preprocessed = np.expand_dims(image_preprocessed, axis=0)
    features = model.predict(image_preprocessed)
    return features.flatten()

def extract_color_histogram(image, mask=None):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image_hsv], [0, 1, 2], mask, [4, 4, 4], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compare_histograms(hist1, hist2):
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity

def extract_region_color_histogram(image, region):
    height, width = image.shape[:2]
    
    if region == 'upper':
        region_img = image[:int(height * 0.5), :]
    elif region == 'lower':
        region_img = image[int(height * 0.5):, :]
    else:
        region_img = image

    return extract_color_histogram(region_img)

all_features = []
image_paths = []
upper_body_count = 0
lower_body_count = 0

for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue

    deep_features = extract_deep_features(image)
    upper_body_hist = extract_region_color_histogram(image, 'upper')
    lower_body_hist = extract_region_color_histogram(image, 'lower')
    upper_vs_lower_similarity = compare_histograms(upper_body_hist, lower_body_hist)
    if upper_vs_lower_similarity < 0.5:
        upper_body_count += 1
    else:
        lower_body_count += 1
    
    all_features.append((deep_features, upper_body_hist, lower_body_hist))
    image_paths.append(img_path)

use_upper_body = upper_body_count > lower_body_count
print(f"Using {'upper' if use_upper_body else 'lower'} body for feature extraction.")

combined_features_list = []
for deep_features, upper_body_hist, lower_body_hist in all_features:
    if use_upper_body:
        combined_histogram = upper_body_hist
    else:
        combined_histogram = lower_body_hist
    
    combined_features = np.hstack([deep_features * 0.5, combined_histogram * 0.5])
    combined_features_list.append(combined_features)

combined_features_array = np.array(combined_features_list)
scaler = StandardScaler()
combined_features_array = scaler.fit_transform(combined_features_array)

start_time = time.time()
clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = clustering.fit_predict(combined_features_array)
elapsed_time = time.time() - start_time
print(f"Clustering took {elapsed_time:.4f} seconds")

def save_image(data):
    img_path, label = data
    if label == 0:
        output_folder = output_folder_player1
    else:
        output_folder = output_folder_player2
    
    img_name = os.path.basename(img_path)
    shutil.copy(img_path, os.path.join(output_folder, img_name))

with ThreadPoolExecutor() as executor:
    executor.map(save_image, zip(image_paths, labels))

total_elapsed_time = time.time() - total_start_time
print(f"Total time for the entire process including saving of images: {total_elapsed_time:.4f} seconds")

print(f"Images saved in {output_folder_player1} and {output_folder_player2}")
