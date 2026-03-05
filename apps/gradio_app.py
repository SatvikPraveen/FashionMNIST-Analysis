"""
Gradio app for FashionMNIST-Analysis.

Lightweight alternative UI for model inference.
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.architectures import ResNet, BasicBlock, MiniCNN, TinyVGG
from src.serving.inference import ImagePreprocessor, RealWorldInference
from src.models.transfer import TransferLearningModel


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


class FashionMNISTPredictor:
    """Predictor class for Gradio interface."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = ImagePreprocessor(target_size=224, device=str(self.device))
        self.model = None
        self.inference = None
        self.current_model = None
    
    def load_model(self, model_name: str):
        """Load specified model."""
        try:
            if model_name == "ResNet":
                self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
            elif model_name == "MiniCNN":
                self.model = MiniCNN(in_channels=1, num_classes=10)
            elif model_name == "TinyVGG":
                self.model = TinyVGG(in_channels=1, hidden_units=32, num_classes=10)
            elif model_name == "Vision Transformer":
                try:
                    tl = TransferLearningModel("vit_base_patch16_224", num_classes=10)
                    self.model = tl.model
                except ImportError:
                    return f"Error: TIMM library not installed"
            
            self.model = self.model.to(self.device).eval()
            self.inference = RealWorldInference(
                model=self.model,
                preprocessor=self.preprocessor,
                device=str(self.device)
            )
            self.current_model = model_name
            return f"✅ {model_name} loaded successfully"
        
        except Exception as e:
            return f"❌ Failed to load model: {e}"
    
    def predict(self, image: Image.Image, model_name: str) -> tuple:
        """Make prediction on image."""
        if model_name != self.current_model:
            status = self.load_model(model_name)
            if "Error" in status or "Failed" in status:
                return None, status
        
        try:
            if image is None:
                return None, "❌ Please upload an image"
            
            image_array = np.array(image.convert("RGB"))
            
            with torch.no_grad():
                prediction = self.inference.predict(image_array, return_top_k=5)
            
            # Format output
            result_text = f"""
### Prediction Results

**Predicted Class:** {prediction['predicted_class_name']}
**Confidence:** {prediction['confidence']:.2%}

#### Top-5 Predictions:
"""
            for i, pred in enumerate(prediction['top_k_predictions'][:5], 1):
                result_text += f"\n{i}. **{pred['class_name']}** - {pred['confidence']:.2%}"
            
            return result_text, "✅ Prediction successful"
        
        except Exception as e:
            return None, f"❌ Prediction failed: {e}"


# Initialize predictor
predictor = FashionMNISTPredictor()


# Create Gradio interface
def make_prediction(image, model_name):
    """Gradio prediction function."""
    result, status = predictor.predict(image, model_name)
    return result, status


# Build interface
with gr.Blocks(title="FashionMNIST Classifier") as demo:
    gr.Markdown(
        """
        # 👕 FashionMNIST Classification
        
        Upload an image to classify fashion items using deep learning models.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            input_image = gr.Image(
                label="Upload Image",
                type="pil"
            )
            
            model_selector = gr.Radio(
                choices=["ResNet", "MiniCNN", "TinyVGG", "Vision Transformer"],
                value="ResNet",
                label="Select Model"
            )
            
            predict_button = gr.Button("🎯 Classify", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### Results")
            output_result = gr.Markdown(
                value="Upload an image and click 'Classify' to see predictions"
            )
            status_message = gr.Textbox(
                label="Status",
                interactive=False
            )
    
    # Connect components
    predict_button.click(
        fn=make_prediction,
        inputs=[input_image, model_selector],
        outputs=[output_result, status_message]
    )
    
    # Example images (optional)
    gr.Examples(
        examples=[],
        inputs=[input_image],
        label="Example Images"
    )
    
    # Information
    gr.Markdown(
        """
        ### Model Information
        - **ResNet**: Residual Network with 18 layers
        - **MiniCNN**: Lightweight custom CNN
        - **TinyVGG**: VGG-inspired architecture
        - **Vision Transformer**: State-of-the-art transformer-based model
        
        ### Fashion MNIST Classes
        1. T-shirt/top
        2. Trouser
        3. Pullover
        4. Dress
        5. Coat
        6. Sandal
        7. Shirt
        8. Sneaker
        9. Bag
        10. Ankle Boot
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
