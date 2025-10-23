# Pneumonia Detection with Streamlit (VS Code Setup)

## 📦 Setup Instructions

### 1. Install Dependencies
Create a virtual environment and install requirements:
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
pip install -r requirements.txt
```

### 2. Train Models (Optional)
Run `train_models.py` to train Custom CNN and ResNet50 models.
Saved models will appear in the `/models` folder as:
- custom_model_final.h5
- resnet_model_final.h5

If you already have trained models, place them in the `models/` folder.

### 3. Run Streamlit App
```bash
streamlit run app.py
```

Then open the displayed local URL in your browser to use the app.

---

## 🧠 Features
- Pneumonia detection using deep learning (Custom CNN + ResNet50)
- Grad-CAM visualization for interpretability
- Streamlit UI for easy image upload and prediction

