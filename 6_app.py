import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title='Brain Tumor Classifier',
    page_icon='🧠',
    layout='centered'
)

# ── Load model ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/efficientnet_best.h5')

model   = load_model()
CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
COLORS  = ['#FF4B4B', '#FFA500', '#00CC00', '#4B8BFF']

# ── Header ───────────────────────────────────────────────────────
st.title('🧠 Brain Tumor MRI Classifier')
st.write('Upload a brain MRI scan and the model will classify the tumor type.')
st.markdown('---')

# ── Sidebar info ─────────────────────────────────────────────────
with st.sidebar:
    st.header('About')
    st.write('This app uses a deep learning model (EfficientNetB0) trained on MRI images to classify brain tumors into 4 categories.')
    st.markdown('---')
    st.subheader('Tumor Classes')
    for cls, col in zip(CLASSES, COLORS):
        st.markdown(f'<span style="color:{col}">● {cls}</span>', unsafe_allow_html=True)
    st.markdown('---')
    st.write('Model: EfficientNetB0 + Transfer Learning')

# ── Upload ───────────────────────────────────────────────────────
uploaded = st.file_uploader(
    'Choose an MRI image',
    type=['jpg', 'jpeg', 'png'],
    help='Upload a brain MRI scan in JPG or PNG format'
)

if uploaded is not None:
    # Show original image
    image = Image.open(uploaded).convert('RGB')
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Uploaded MRI Scan')
        st.image(image, use_container_width=True)

    # Preprocess
    img_array = np.array(image.resize((224, 224))) / 255.0
    x         = img_array[np.newaxis, ...]

    # Predict
    with st.spinner('Analyzing MRI scan...'):
        predictions = model.predict(x, verbose=0)[0]
        pred_index  = np.argmax(predictions)
        pred_label  = CLASSES[pred_index]
        confidence  = predictions[pred_index] * 100

    with col2:
        st.subheader('Prediction Result')
        st.markdown(
            f'<div style="background-color:{COLORS[pred_index]}22;'
            f'border-left: 4px solid {COLORS[pred_index]};'
            f'padding:16px;border-radius:8px;">'
            f'<h2 style="color:{COLORS[pred_index]};margin:0">{pred_label}</h2>'
            f'<p style="margin:4px 0 0">Confidence: <b>{confidence:.1f}%</b></p>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown('<br>', unsafe_allow_html=True)

        # Confidence bar for each class
        st.subheader('Class Probabilities')
        for i, (cls, prob) in enumerate(zip(CLASSES, predictions)):
            st.markdown(f'**{cls}**')
            st.progress(float(prob))
            st.caption(f'{prob*100:.2f}%')

    st.markdown('---')

    # Grad-CAM section
    st.subheader('Grad-CAM Heatmap')
    st.write('Shows which regions of the MRI the model focused on to make its prediction.')

    try:
        img_cv   = np.array(image.resize((224, 224)))
        img_norm = img_cv / 255.0
        x_tf     = img_norm[np.newaxis, ...]

        grad_model = tf.keras.Model(
            model.inputs,
            [model.get_layer('top_conv').output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x_tf)
            loss = preds[:, pred_index]

        grads   = tape.gradient(loss, conv_out)[0]
        cam     = tf.reduce_mean(grads, axis=(0, 1)).numpy()
        cam     = np.maximum(cam, 0)
        cam     = cam / (cam.max() + 1e-8)
        cam     = cv2.resize(cam, (224, 224))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

        col3, col4 = st.columns(2)
        with col3:
            st.image(heatmap, caption='Heatmap', use_container_width=True)
        with col4:
            st.image(overlay, caption='Overlay on MRI', use_container_width=True)

    except Exception as e:
        st.warning(f'Grad-CAM could not be generated: {e}')

else:
    # Placeholder when no image uploaded
    st.info('👆 Upload an MRI image above to get started.')
    st.markdown('---')
    st.subheader('How it works')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('### 1️⃣ Upload')
        st.write('Upload any brain MRI scan in JPG or PNG format')
    with col2:
        st.markdown('### 2️⃣ Analyze')
        st.write('Model analyzes the scan using deep learning')
    with col3:
        st.markdown('### 3️⃣ Result')
        st.write('Get tumor classification with confidence score')