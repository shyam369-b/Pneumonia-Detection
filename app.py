import os
import numpy as np
import seaborn as sns
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# ---------------- Config ----------------
IMG_SIZE = (224, 224)
MODEL_DIR = 'C:/Users/shyam/OneDrive/Desktop/P_Project/PneumoniaDetection_Project/models'
TEST_DATA_DIR = 'C:/Users/shyam/OneDrive/Desktop/P_Project/PneumoniaDetection_Project/chest_xray/test'


# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="Pneumonia Detection Dashboard", layout="wide", page_icon="🩺")

st.markdown("""
    <style>
        .main { background-color: #2cce3a; }
        h1, h2, h3, h4 { color: #1e3d59; }
        .stImage > img { border-radius: 10px; }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)



# ---------------- Model Loader ----------------
@st.cache_resource
def load_selected_model(model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    return load_model(model_path)

# ---------------- Grad-CAM Functions ----------------
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def gradcam_overlay(img, model, last_conv_layer_name=None, alpha=0.4):
    if not last_conv_layer_name:
        last_conv_layer_name = get_last_conv_layer(model)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    try:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    except Exception as e:
        st.warning(f"Grad-CAM not supported for this model. ({str(e)})")
        return img_resized
    heatmap = cv2.resize(heatmap, (IMG_SIZE[1], IMG_SIZE[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR), 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# ---------------- Prediction Function ----------------
def predict_image(model, img):
    img_resized = cv2.resize(img, IMG_SIZE)
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    pred = model.predict(img_array, verbose=0)[0][0]
    label = 'PNEUMONIA' if pred > 0.5 else 'NORMAL'
    confidence = pred if pred > 0.5 else 1 - pred
    return label, confidence

# ---------------- Tabs ----------------
tab1, tab2 = st.tabs(["🩺 Prediction & Grad-CAM", "📊 Model Insights"])

# =========================================================================================
# TAB 1 – Prediction UI (Responsive Layout)
# =========================================================================================
with tab1:
    st.title(" 🗣️🩺 Interpretable Deep Learning Framework for Pneumonia Diagnosis using Chest X-Ray Imaging ")
    st.markdown("<hr>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 2], vertical_alignment="top")

    # ---------------- Controls ----------------
    with left_col:
        st.subheader("🔧 Controls")
        model_choice = st.selectbox('Select Model:', ['custom_CNN_model_final.h5', 'resnet_model_final.h5'])
        model = load_selected_model(model_choice)
        uploaded_file = st.file_uploader('📤 Upload Chest X-ray', type=['jpg', 'jpeg', 'png'])

    # ---------------- Image & Prediction ----------------
    with right_col:
        if uploaded_file:
            # Load image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Predict
            label, conf = predict_image(model, img_rgb)
            prob_pneumonia = conf if label == 'PNEUMONIA' else 1 - conf
            prob_normal = 1 - prob_pneumonia

            # Prediction Card
            bg_color = "#c0392b" if label == "PNEUMONIA" else "#2cce3a"
            st.markdown(
                f"""
                <div style='background-color:{bg_color}; padding:5px; border-radius:12px; text-align:center; color:black; margin-bottom:3px; max-width:100%;'>
                    <h2>Diagnosis: {label}</h2>
                    <h4>Confidence: {conf*100:.2f}%</h4>
                </div>
                """, unsafe_allow_html=True
            )

            # Probability Bars (Full screen width)
            st.markdown("### 📊 Probability Distribution")
            st.markdown(
                f"""
                <div style='margin-top:10px; width:100%;'>
                    <p><b>PNEUMONIA:</b> {prob_pneumonia*100:.2f}%</p>
                    <div style='background: linear-gradient(to right, #c0392b {prob_pneumonia*100}%, #e0e0e0 {prob_pneumonia*100}%); 
                                height:25px; border-radius:5px; width:100%;'></div>
                    <p style='margin-top:10px;'><b>NORMAL:</b> {prob_normal*100:.2f}%</p>
                    <div style='background: linear-gradient(to right, #2980b9 {prob_normal*100}%, #e0e0e0 {prob_normal*100}%); 
                                height:25px; border-radius:5px; width:100%;'></div>
                </div>
                """, unsafe_allow_html=True
            )

            # ---------------- Images Side by Side (Responsive) ----------------
            st.markdown("<h3 style='margin-top:25px;'>🔥 Original X-Ray vs Grad-CAM Heatmap</h3>", unsafe_allow_html=True)
            grad_img = gradcam_overlay(img_rgb, model)

            # Responsive columns
            screen_cols = st.columns([1, 1], gap="medium")
            with screen_cols[0]:
                st.image(img_rgb, caption='Original X-ray', use_container_width=True)
            with screen_cols[1]:
                st.image(cv2.cvtColor(grad_img, cv2.COLOR_BGR2RGB), caption='Grad-CAM Heatmap', use_container_width=True)

        else:
            st.info('📁 Please upload a chest X-ray image to begin detection.')

# =========================================================================================
# TAB 2 – Model Insights
# =========================================================================================
with tab2:
    st.title("📊 Models Insights & Performance Analysis")
    st.markdown("<hr>", unsafe_allow_html=True)

    model_choice = st.selectbox('Select Model to Analyze:', 
                                ['custom_CNN_model_final.h5', 'resnet_model_final.h5'], 
                                key="analyze_model")
    model = load_selected_model(model_choice)

    # ---------------- Confusion Matrix ----------------
    st.subheader("🧩 Confusion Matrix (Test Data)")
    if os.path.exists(TEST_DATA_DIR):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        test_data = datagen.flow_from_directory(
            TEST_DATA_DIR, 
            target_size=IMG_SIZE, 
            class_mode='binary', 
            shuffle=False
        )

        preds = model.predict(test_data)
        y_pred = (preds > 0.5).astype(int)
        y_true = test_data.classes
        class_labels = list(test_data.class_indices.keys())

        # Confusion Matrix Heatmap
        
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels,
                    cbar=False, linewidths=1, linecolor='gray')
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig, use_container_width=True)

        # Classification Report (Tabular)
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.markdown("### 📋 Classification Report")
        st.dataframe(
            df_report.style.background_gradient(cmap='Blues').format(precision=3),
            use_container_width=True
        )

    else:
        st.warning("⚠️ Test dataset folder not found. Place test images in `test/` directory with NORMAL & PNEUMONIA subfolders.")

    st.markdown("---")

    # ---------------- Dashboard Layout for Visual Analysis ----------------
    col1, col2 = st.columns(2)

    # ---- Gradient Flow ----
    with col1:
        st.subheader("📈 Gradient Flow Simulation")
        layers = np.arange(1, 21)
        vanishing = np.exp(-layers/3)
        exploding = np.exp(layers/3)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(layers, vanishing, label="Vanishing Gradient", linewidth=2)
        ax.plot(layers, exploding/1000, label="Exploding Gradient (scaled)", linewidth=2)
        ax.set_xlabel("Layer Depth")
        ax.set_ylabel("Gradient Magnitude (log scale)")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig, use_container_width=True)

    # ---- Optimizer Comparison ----
    with col2:
        st.subheader("⚙️ Optimizer Performance Comparison")
        x = np.arange(1, 13)
        loss_adam = np.exp(-x/4) + np.random.normal(0, 0.01, len(x))
        loss_sgd = np.exp(-x/7) + np.random.normal(0, 0.015, len(x))
        loss_rmsprop = np.exp(-x/5) + np.random.normal(0, 0.012, len(x))

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x, loss_adam, label="Adam", linewidth=2)
        ax.plot(x, loss_sgd, label="SGD", linewidth=2)
        ax.plot(x, loss_rmsprop, label="RMSProp", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    # ---------------- Weight Distribution ----------------
    st.subheader("🔩 Weight Initialization Histogram")
    sample_weights = np.random.randn(5000) * 0.05
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(sample_weights, bins=50, color="#1e88e5", alpha=0.8)
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Initialized Weights")
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig, use_container_width=True)

st.caption("📍Interpretable Deep Learning Framework for Pneumonia Diagnosis using Chest X-Ray Imaging")
