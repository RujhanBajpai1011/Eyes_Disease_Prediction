import streamlit as st
import tensorflow.compat.v1 as tf
import tf_keras as keras
import numpy as np
from PIL import Image
import recommendation as rec
import requests
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RetinaNet - OCT Analysis",
    page_icon="👁️",
    layout="centered", # Reverted to centered for better focus
    initial_sidebar_state="expanded",
)

# --- CLASSIC STYLING ---
st.markdown("""
    <style>
    .stButton>button {
        background-color: #f0f2f6;
        color: #31333F;
        border: 1px solid #dcdfe4;
    }
    .prediction-text {
        font-size: 24px;
        font-weight: bold;
        color: #007BFF;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CORE FUNCTIONS ---
MODEL_URL = "https://github.com/RujhanBajpai1011/Eyes_Disease_Prediction/releases/download/v1.0.0/Trained_Model.h5"
MODEL_PATH = "Trained_Model.h5"

@st.cache_resource # Taaki model baar-baar download na ho
def load_retinanet_model():
    # Agar file local folder mein nahi hai, toh download karo
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from GitHub Releases... Please wait."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    
    return keras.models.load_model(MODEL_PATH)
# Model load karein
model = load_retinanet_model()

# Class order matches Training_model.ipynb categories
CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("👁️ RetinaNet")
    st.write("---")
    page = st.radio(
        "Navigation", 
        ["🏠 Home", "🔮 Prediction", "🔬 Technical Details", "📖 Disease Library"],
        index=0
    )
    st.write("---")
    st.info("Medical Diagnostic Support Tool")

# --- PAGE: HOME ---
if page == "🏠 Home":
    st.title("RetinaNet: AI Retinal Analysis")
    st.markdown("""
    Welcome to the RetinaNet platform. This tool is designed to provide rapid classification of **Optical Coherence Tomography (OCT)** retinal scans using Deep Learning.
    
    #### How it works:
    1. **Upload**: Provide a cross-sectional retinal scan.
    2. **Analysis**: The system uses a **MobileNetV3Large** neural network.
    3. **Results**: Receive a diagnosis and medical recommendations.
    """)
    if st.button("Get Started"):
        st.write("Please select 'Prediction' from the sidebar menu.")

# --- PAGE: PREDICTION (Alignment Fixed) ---
elif page == "🔮 Prediction":
    st.title("Scan Analysis")
    st.write("Upload an OCT image to begin.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        # 1. Show the Image First
        st.image(image, caption="Uploaded OCT Scan", use_container_width=True)
        
        with st.spinner("Analyzing scan..."):
            # Preprocessing to match training (224, 224, 3)
            processed_img = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(processed_img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            preds = model.predict(img_array)
            label = CLASS_NAMES[np.argmax(preds)]
            confidence = np.max(tf.nn.softmax(preds[0])) * 100

        # 2. Show Prediction directly below
        st.markdown(f"**Analysis Result:** <span class='prediction-text'>{label}</span>", unsafe_allow_html=True)
        st.write(f"Confidence: {confidence:.2f}%")
        
        st.write("---")
        
        # 3. Show Recommendations directly below the prediction
        st.subheader("Clinical Recommendation")
        # Fetches localized advice from recommendation.py
        if label == 'CNV': st.markdown(rec.cnv)
        elif label == 'DME': st.markdown(rec.dme)
        elif label == 'DRUSEN': st.markdown(rec.drusen)
        else: st.markdown(rec.normal)

# --- PAGE: TECHNICAL DETAILS ---
elif page == "🔬 Technical Details":
    st.title("Model & Training Technicalities")
    st.write("This model was developed using the following specifications:")
    
    st.markdown("""
    * **Model Type**: MobileNetV3Large (Transfer Learning)
    * **Training Dataset**: 76,515 OCT images across 4 classes
    * **Optimizer**: Adam (Learning Rate: 0.0001)
    * **Input Resolution**: 224x224 pixels
    * **Epochs**: 15
    """)
    st.success("The final model achieved a test accuracy of approximately 95%.")

# --- PAGE: DISEASE LIBRARY ---
elif page == "📖 Disease Library":
    st.title("OCT Disease Glossary")
    st.write("Educational content regarding the conditions identified:")
    
    with st.expander("Choroidal Neovascularization (CNV)"):
        st.write("Growth of new blood vessels beneath the retina, often requiring anti-VEGF therapy.")
        
    with st.expander("Diabetic Macular Edema (DME)"):
        st.write("Fluid accumulation in the macula as a complication of diabetes.")
        
    with st.expander("Drusen"):
        st.write("Lipid deposits under the retina associated with age-related macular degeneration.")



