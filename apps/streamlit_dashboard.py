"""
Streamlit dashboard for FashionMNIST-Analysis.

Interactive dashboard for model inference, comparison, and visualization.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.model_definitions import ResNet, BasicBlock, MiniCNN, TinyVGG
from src.transfer_learning import TransferLearningModel
from src.real_world_inference import ImagePreprocessor, RealWorldInference
from src.ensemble import EnsembleVoting
from src.explainability import GradCAM
import matplotlib.pyplot as plt


# Page configuration
st.set_page_config(
    page_title="FashionMNIST Analysis",
    page_icon="👕",
    layout="wide"
)

st.title("🎨 FashionMNIST Classification Dashboard")
st.markdown("---")


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    mode = st.radio(
        "Select Mode",
        ["Single Prediction", "Batch Prediction", "Model Comparison", "Explainability"]
    )
    
    model_type = st.selectbox(
        "Model Type",
        ["ResNet", "MiniCNN", "TinyVGG", "Vision Transformer"]
    )
    
    if model_type == "Vision Transformer":
        st.info("ℹ️ Vision Transformer requires TIMM library")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        0.0, 1.0, 0.5, 0.05
    )
    
    top_k = st.slider(
        "Top-K Predictions",
        1, 10, 3, 1
    )


# Class names
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]


@st.cache_resource
def load_model(model_type: str, device: str = "cpu"):
    """Load model with caching."""
    try:
        if model_type == "ResNet":
            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        elif model_type == "MiniCNN":
            model = MiniCNN(in_channels=1, num_classes=10)
        elif model_type == "TinyVGG":
            model = TinyVGG(in_channels=1, hidden_units=32, num_classes=10)
        elif model_type == "Vision Transformer":
            try:
                tl = TransferLearningModel("vit_base_patch16_224", num_classes=10)
                model = tl.model
            except ImportError:
                st.error("TIMM library not installed")
                return None
        
        model.to(device).eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


@st.cache_resource
def get_preprocessor():
    """Get image preprocessor."""
    return ImagePreprocessor(target_size=224)


# Single Prediction Mode
if mode == "Single Prediction":
    st.header("📸 Single Image Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            image_array = np.array(image)
        else:
            image_array = None
    
    with col2:
        st.subheader("Prediction Results")
        
        if image_array is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load model and preprocessor
            model = load_model(model_type, device)
            preprocessor = get_preprocessor()
            
            if model:
                inference = RealWorldInference(
                    model=model,
                    preprocessor=preprocessor,
                    device=device,
                    confidence_threshold=confidence_threshold
                )
                
                with st.spinner("Making prediction..."):
                    prediction = inference.predict(image_array, return_top_k=top_k)
                
                # Display main prediction
                st.metric(
                    "Predicted Class",
                    prediction["predicted_class_name"],
                    f"{prediction['confidence']:.2%}"
                )
                
                # Display top-K
                st.subheader("Top Predictions")
                for i, pred in enumerate(prediction["top_k_predictions"], 1):
                    col_label, col_conf = st.columns([3, 1])
                    with col_label:
                        st.write(f"{i}. {pred['class_name']}")
                    with col_conf:
                        st.write(f"{pred['confidence']:.2%}")
                
                # Display probability distribution
                st.subheader("Class Probability Distribution")
                probs_df = {
                    "Class": list(prediction["all_probabilities"].keys()),
                    "Probability": list(prediction["all_probabilities"].values())
                }
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(probs_df["Class"], probs_df["Probability"])
                ax.set_xlabel("Probability")
                ax.set_title("Class Probabilities")
                st.pyplot(fig)


# Batch Prediction Mode
elif mode == "Batch Prediction":
    st.header("📁 Batch Image Prediction")
    
    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(model_type, device)
        preprocessor = get_preprocessor()
        
        if model:
            with st.spinner("Processing images..."):
                images = []
                predictions_list = []
                
                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file).convert("RGB")
                    images.append(np.array(image))
                
                inference = RealWorldInference(
                    model=model,
                    preprocessor=preprocessor,
                    device=device,
                    confidence_threshold=confidence_threshold
                )
                
                predictions_list = inference.predict_batch(images, return_top_k=1)
            
            # Display results
            st.subheader(f"Results ({len(predictions_list)} images)")
            
            cols = st.columns(3)
            for idx, (file, pred) in enumerate(zip(uploaded_files, predictions_list)):
                with cols[idx % 3]:
                    image = Image.open(file).convert("RGB")
                    st.image(image, use_column_width=True)
                    
                    pred_class = pred["top_k_predictions"][0]["class_name"]
                    confidence = pred["top_k_predictions"][0]["confidence"]
                    
                    st.success(f"**{pred_class}**\n{confidence:.2%}")


# Model Comparison Mode
elif mode == "Model Comparison":
    st.header("🔄 Model Comparison")
    
    st.info("📌 This mode compares predictions from multiple models on the same image")
    
    uploaded_file = st.file_uploader(
        "Upload an image for comparison",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Test Image", use_column_width=True)
        image_array = np.array(image)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        preprocessor = get_preprocessor()
        
        # Compare models
        models_to_compare = ["ResNet", "MiniCNN", "TinyVGG"]
        
        st.subheader("Predictions")
        
        comparison_data = []
        
        for model_name in models_to_compare:
            model = load_model(model_name, device)
            
            if model:
                inference = RealWorldInference(
                    model=model,
                    preprocessor=preprocessor,
                    device=device
                )
                
                prediction = inference.predict(image_array, return_top_k=1)
                
                comparison_data.append({
                    "Model": model_name,
                    "Prediction": prediction["predicted_class_name"],
                    "Confidence": f"{prediction['confidence']:.2%}"
                })
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**{model_name}**")
                with col2:
                    st.write(prediction["predicted_class_name"])
                with col3:
                    st.write(f"{prediction['confidence']:.2%}")
        
        # Comparison table
        st.subheader("Comparison Summary")
        st.table(comparison_data)


# Explainability Mode
elif mode == "Explainability":
    st.header("🔍 Model Explainability")
    
    st.info("📌 This mode provides interpretability insights into model predictions")
    
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Input Image", use_column_width=True)
        image_array = np.array(image)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(model_type, device)
        preprocessor = get_preprocessor()
        
        if model:
            # Make prediction
            inference = RealWorldInference(
                model=model,
                preprocessor=preprocessor,
                device=device
            )
            
            prediction = inference.predict(image_array, return_top_k=3)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction")
                st.metric(
                    "Predicted Class",
                    prediction["predicted_class_name"],
                    f"{prediction['confidence']:.2%}"
                )
            
            with col2:
                st.subheader("Top-3 Predictions")
                for pred in prediction["top_k_predictions"]:
                    st.write(f"• {pred['class_name']}: {pred['confidence']:.2%}")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: small;">
    FashionMNIST Analysis Dashboard | Powered by Streamlit & PyTorch
    </div>
    """,
    unsafe_allow_html=True
)


if __name__ == "__main__":
    pass
