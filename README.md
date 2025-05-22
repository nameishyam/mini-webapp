# 🧠 DR Analyzer — Diabetic Retinopathy Classification Web App

**DR Analyzer** is a lightweight, Flask-based web application that classifies retinal fundus images into different stages of **Diabetic Retinopathy (DR)**. Designed for clinical insight and academic utility, it leverages a deep learning model trained on real-world datasets to provide accurate and immediate feedback.

> ⚕️ Empowering healthcare professionals and researchers with AI-driven diagnostics for retinal health.

---

## 🚀 Features

- Upload and analyze fundus images directly through the browser.
- Classifies into five stages: **No DR**, **Mild**, **Moderate**, **Severe**, and **Proliferative DR**.
- Real-time inference via a lightweight Flask backend.
- Simple, intuitive web interface.
- Easily extendable to support new models or additional ophthalmic conditions.

---

## 🧠 Model Overview

- Based on a custom **DRNet architecture**, optimized for retinal image analysis.

---

## 🐳 Docker Deployment

### Prerequisites

- Docker installed on your system
- Docker Compose installed on your system

### Building and Running with Docker

1. **Build and start the container:**

   ```bash
   docker-compose up -d
   ```

2. **Access the application:**
   Open your browser and go to:

   ```
   http://localhost:5000
   ```

3. **Stop the container:**
   ```bash
   docker-compose down
   ```

### Building without Docker Compose

If you prefer not to use Docker Compose:

1. **Build the Docker image:**

   ```bash
   docker build -t dr-analyzer .
   ```

2. **Run the container:**

   ```bash
   docker run -d -p 5000:5000 --name dr-analyzer dr-analyzer
   ```

3. **Stop and remove the container:**
   ```bash
   docker stop dr-analyzer
   docker rm dr-analyzer
   ```

- Trained on large-scale datasets like **APTOS** and **EyePACS**.
- Includes preprocessing (grayscale normalization, contrast enhancement, resizing).
- Utilizes multi-scale features and attention mechanisms.
- Optional use of perceptual loss for enhanced visual consistency in image-driven diagnosis.

---

## 📁 Repository Structure

```

├── static/
├── templates/
├── model/
├── app.py
├── requirements.txt
├── README.md
├── CONTRIBUTIONS.md
├── CODE_OF_CONDUCT.md

```

---

## ⚙️ Getting Started

Run **DR Analyzer** locally by following the steps below:

### 1. Clone the Repository

```bash
git clone https://github.com/nameishyam/mini-webapp.git
cd mini-webapp
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Add Pretrained Model

Download or place your trained model (e.g., `drnet_model.pth`) into the `model/` folder.

> ⚠️ The model file is not included due to size limits. Contact the author or train your own using the APTOS/EyePACS dataset.

### 5. Launch the Application

```bash
python app.py
```

Visit the app at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧪 Sample Usage

Upload retinal fundus images in `.jpg`, `.jpeg`, or `.png` format. The prediction will appear immediately after upload, along with class labels.

---

## 📊 Roadmap

- ✅ Core classification via DRNet
- 🚧 Grad-CAM-based visual explanations
- 🚧 Deployment via Docker and CI/CD
- 🚧 REST API version for mobile and external use

---

## 🙌 How to Contribute

We welcome and value contributions from the community. Whether it's improving the UI, enhancing the model, fixing bugs, or writing documentation—your support matters.

- Review the [📘 CONTRIBUTIONS.md](CONTRIBUTING.md) for guidelines.
- Please follow our [🤝 CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) to maintain a respectful and inclusive environment.

---

## 📄 License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## 📬 Contact

- GitHub: [@nameishyam](https://github.com/nameishyam)
- Email: geddamgowtham4@gmail.com
