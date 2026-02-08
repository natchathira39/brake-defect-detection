import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Page config
st.set_page_config(
    page_title="Brake Defect Detection",
    page_icon="🔧",
    layout="centered"
)

# Configuration
CLASS_NAMES = ['good', 'patches', 'rolled_pits', 'scratches', 'waist_folding']
IMG_SIZE = (224, 224)

# Find model file
@st.cache_resource
def load_trained_model():
    # Find any .h5 file in the directory
    model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
    if not model_files:
        st.error("❌ No model file found! Please upload a .h5 file.")
        return None
    
    model_file = model_files[0]
    st.info(f"📦 Loading model: {model_file}")
    
    try:
        model = load_model(model_file)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_trained_model()

# Title and description
st.title("🔧 Automotive Disc Brake Defect Detection")
st.write("Upload an image of a brake disc to detect defects using deep learning.")

st.markdown("""
**Defect Classes:**
- ✅ **good** - No defects detected
- ⚠️ **patches** - Surface patches detected
- ⚠️ **rolled_pits** - Rolling pits detected
- ⚠️ **scratches** - Scratches detected
- ⚠️ **waist_folding** - Waist folding detected
""")

# File uploader
uploaded_file = st.file_uploader("📤 Choose a brake disc image...", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None and model is not None:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Predict button
    if st.button('🔍 Analyze Defects', type="primary", use_container_width=True):
        with st.spinner('🔄 Analyzing image...'):
            try:
                # Preprocess
                img_resized = image.resize(IMG_SIZE)
                img_array = img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0
                
                # Predict
                predictions = model.predict(img_array, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                predicted_class = CLASS_NAMES[predicted_class_idx]
                confidence = float(predictions[0][predicted_class_idx])
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    status = "✅ Good" if predicted_class == 'good' else "⚠️ Defective"
                    st.metric("Status", status)
                
                with col2:
                    st.metric("Defect Type", predicted_class)
                
                with col3:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Show all probabilities
                st.subheader("📊 Class Probabilities")
                
                for class_name, prob in zip(CLASS_NAMES, predictions[0]):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.progress(float(prob))
                    with col2:
                        st.write(f"**{class_name}**: {prob:.2%}")
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
