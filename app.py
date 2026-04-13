from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import os
import uuid
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

app = Flask(__name__)

# 📁 Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🧠 Load model
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# 📊 Load dataset features
features_list = np.load("features.npy")
image_paths = np.load("paths.npy")

# 📚 Fish descriptions
descriptions = {
    "moonfish": "Flat, silver fish with round body, commonly found in coastal waters.",
    "nile_tilapia": "Freshwater fish with flat body, widely farmed and commonly eaten.",
    "parrotfish": "Colorful reef fish with beak-like mouth, found in coral reefs.",
    "mahi-mahi": "Fast-swimming ocean fish with bright colors, also called dolphinfish.",
    "yellowstripe_scad": "Small silver fish with yellow stripe, usually found in schools.",
    "red_tilapia": "Reddish freshwater fish, similar to tilapia but with pink to red color.",
    "barracuda": "Long, slender fish with sharp teeth, known as a fast predator."
}

# 🔍 Feature extraction
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img, verbose=0)
    return features.flatten()

# 🏠 Home
@app.route('/')
def home():
    return render_template('index.html')

# 📤 Upload page
@app.route('/upload_page')
def upload_page():
    return render_template('upload.html')

# 📥 Upload handler
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']

    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    return redirect(url_for('loading', filename=filename))

# ⏳ Loading page (FIXED)
@app.route('/loading/<filename>')
def loading(filename):
    return render_template(
        "loading.html",
        redirect_url=url_for("result", filename=filename)
    )

# 📊 Result page (FIXED)
@app.route('/result/<filename>')
def result(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # 🔍 Extract features
    query_features = extract_features(filepath).reshape(1, -1)

    # 📊 Similarity
    similarities = cosine_similarity(query_features, features_list)[0]

    # 🔥 Top 3
    top_indices = similarities.argsort()[-3:][::-1]

    results = []
    scores = []

    for i in top_indices:
        results.append(image_paths[i])
        scores.append(round(similarities[i] * 100, 2))

    # 🧠 Label
    labels = [os.path.basename(os.path.dirname(path)) for path in results]
    final_label = Counter(labels).most_common(1)[0][0]
    final_label = final_label.lower().strip()

    # 🎯 Confidence
    confidence = max(scores)

    # 📚 Description
    description = descriptions.get(final_label, "No description available.")

    # 🎨 Format label
    display_label = final_label.replace("_", " ").title()

    print("RESULT PATHS:", results)
    print("LABELS:", labels)
    print("FINAL:", final_label)

    return render_template(
        'result.html',
        query=filepath,
        results=results,
        scores=scores,
        label=display_label,
        confidence=confidence,
        description=description,
        labels=labels
        
    )

# 🚀 Run
if __name__ == "__main__":
    app.run(debug=True)