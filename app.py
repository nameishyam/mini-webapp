from flask import Flask, render_template, request, jsonify
import os
import torch
from torchvision import transforms
from PIL import Image
from werkzeug.utils import secure_filename
import torch.nn as nn
from torchvision.models import vgg16

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = './models/model.pth'  # Ensure this path matches where your model is stored

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device setup: use GPU if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your custom VGG16 architecture to match the saved model state_dict
class CustomVGG16(nn.Module):
    def __init__(self):
        super(CustomVGG16, self).__init__()
        self.features = vgg16(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Function to load the model using the appropriate device mapping
def load_model():
    try:
        model = CustomVGG16()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()  # Set model to evaluation mode
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define image transformation consistent with model training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Preprocess the image for model input
            image = Image.open(filepath).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            # Run prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)

            # For this model, predicted index corresponds to class label (0 to 4)
            class_idx = predicted.item()
            prediction = f"{class_idx}"

            return jsonify({
                'success': True,
                'filename': filename,
                'prediction': prediction,
                'message': f'File {filename} uploaded and classified successfully'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug = True)
