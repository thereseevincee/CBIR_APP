import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(224, 224, 3)
)

dataset_path = "static/images"

features_list = []
image_paths = []

print("Extracting features...")

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(root, file)

            img = image.load_img(path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            features = model.predict(img, verbose=0)

            features_list.append(features.flatten())
            image_paths.append(path)

features_list = np.array(features_list, dtype="float32")

print("Feature shape:", features_list.shape)

np.save("features.npy", features_list)
np.save("paths.npy", np.array(image_paths))

print("DONE ✅")