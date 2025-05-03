import os
import torch
from flask import Flask, request, jsonify, render_template
import uuid
import json
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision.models import resnet18
from inference_sdk import InferenceHTTPClient

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="uMnBoh9m5MD03z98rAdO"
)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.backbone = resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.fc(features)
        return embeddings

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_siamese_model():
    model = SiameseNetwork()
    model.load_state_dict(torch.load("models/siamese_best_model.pth", map_location='cpu'))
    model.eval()
    return model

def l2_normalize(embedding):
    norm = torch.norm(embedding, p=2, dim=-1, keepdim=True)
    return embedding / norm

recognition_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

metadata = "metadata.json"
siamese_model = load_siamese_model()

def process_image(image_path, shrink_ratio=0.2):
    result = CLIENT.infer(image_path, model_id="muzzle-detection-h7o8o/1")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    crops = []
    for pred in result['predictions']:
        new_width = max(1, int(pred['width'] * (1 - shrink_ratio)))
        new_height = max(1, int(pred['height'] * (1 - shrink_ratio)))
        center_x, center_y = int(pred['x']), int(pred['y'])
        x1 = max(center_x - new_width // 2, 0)
        y1 = max(center_y - new_height // 2, 0)
        x2 = min(center_x + new_width // 2, image.shape[1])
        y2 = min(center_y + new_height // 2, image.shape[0])
        
        # Draw bounding box
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cropped = image_rgb[y1:y2, x1:x2]
        crop_filename = f"crop_{uuid.uuid4().hex}.jpg"
        crop_path = os.path.join(app.config['UPLOAD_FOLDER'], crop_filename)
        cv2.imwrite(crop_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        crops.append(crop_filename)

    annotated_filename = f"annotated_{uuid.uuid4().hex}.jpg"
    annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
    cv2.imwrite(annotated_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return annotated_filename, crops

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register_cattle():
    files = request.files.getlist('files[]')
    if len(files) != 5:
        return jsonify({"error": "Exactly 5 images required"}), 400

    cattle_id = request.form['cattle_id']
    embeddings = []

    for file in files:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}.jpg")
        file.save(img_path)
        annotated_img, crops = process_image(img_path)
        
        if not crops:
            continue
            
        crop_path = os.path.join(app.config['UPLOAD_FOLDER'], crops[0])
        img_pil = Image.open(crop_path).convert("RGB")
        img_tensor = recognition_transform(img_pil).unsqueeze(0)
        
        with torch.no_grad():
            embedding = siamese_model(img_tensor)
            embedding = l2_normalize(embedding).squeeze(0).numpy().tolist()
            embeddings.append(embedding)

    if len(embeddings) < 5:
        return jsonify({"error": "Could not detect muzzle in all images"}), 400

    # Calculate mean embedding
    mean_embedding = np.mean(embeddings, axis=0).tolist()

    embeddings_db = {}
    if os.path.exists(metadata):
        with open(metadata, 'r') as f:
            embeddings_db = json.load(f)

    embeddings_db[cattle_id] = mean_embedding

    with open(metadata, 'w') as f:
        json.dump(embeddings_db, f)

    return jsonify({
        "message": f"Cattle {cattle_id} registered!",
        "annotated_img": annotated_img,
        "crops": crops
    })

@app.route('/recognize', methods=['POST'])
def recognize_cattle():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}.jpg")
    file.save(img_path)
    annotated_img, crops = process_image(img_path)

    if not crops:
        return jsonify({"error": "No muzzle detected"}), 400

    crop_path = os.path.join(app.config['UPLOAD_FOLDER'], crops[0])
    img_pil = Image.open(crop_path).convert("RGB")
    img_tensor = recognition_transform(img_pil).unsqueeze(0)

    with torch.no_grad():
        embedding = siamese_model(img_tensor)
        embedding = l2_normalize(embedding).squeeze(0).numpy()

    embeddings_db = {}
    if os.path.exists(metadata):
        with open(metadata, 'r') as f:
            embeddings_db = json.load(f)

    max_similarity = -1
    recognized_cattle_id = None

    for cattle_id, stored_embedding in embeddings_db.items():
        similarity = 1 - cosine(embedding, stored_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            recognized_cattle_id = cattle_id

    threshold = 0.85  # Set a threshold for recognition
    if max_similarity >= threshold:
        return jsonify({
            "recognized_cattle_id": recognized_cattle_id,
            "similarity": max_similarity,
            "annotated_img": annotated_img,
            "crops": crops
        })
    
    return jsonify({
        "message": "Cattle not recognized",
        "similarity": max_similarity,
        "annotated_img": annotated_img,
        "crops": crops
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)