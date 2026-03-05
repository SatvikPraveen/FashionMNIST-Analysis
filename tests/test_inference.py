"""
Unit tests for inference utilities.
"""

import pytest
import torch
import numpy as np
from PIL import Image
import io


class TestRealWorldInference:
    """Test real-world inference utilities."""
    
    def test_image_preprocessor_init(self):
        """Test ImagePreprocessor initialization."""
        from src.serving.inference import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(
            target_size=224,
            normalize=True,
            convert_to_rgb=True
        )
        
        assert preprocessor.target_size == 224
        assert preprocessor.normalize is True
    
    def test_image_preprocessing(self, tmp_path):
        """Test image preprocessing."""
        from src.serving.inference import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(target_size=224)
        
        # Create dummy image
        img = Image.new('RGB', (256, 256), color='red')
        img_path = tmp_path / "test_image.png"
        img.save(img_path)
        
        # Preprocess
        processed = preprocessor.preprocess(img_path, return_tensor=True)
        
        assert isinstance(processed, torch.Tensor)
        assert processed.shape == (1, 3, 224, 224)
    
    def test_grayscale_to_rgb_conversion(self):
        """Test grayscale to RGB conversion."""
        from src.serving.inference import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(target_size=224, convert_to_rgb=True)
        
        # Create grayscale image
        gray_img = np.ones((100, 100), dtype=np.uint8) * 128
        
        processed = preprocessor.preprocess(gray_img, return_tensor=True)
        
        # Should have 3 channels (RGB)
        assert processed.shape[1] == 3
    
    def test_batch_preprocessing(self, tmp_path):
        """Test batch preprocessing."""
        from src.serving.inference import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(target_size=224)
        
        # Create dummy images
        image_paths = []
        for i in range(3):
            img = Image.new('RGB', (256, 256), color=('red', 'green', 'blue')[i])
            img_path = tmp_path / f"test_image_{i}.png"
            img.save(img_path)
            image_paths.append(img_path)
        
        # Batch preprocess
        batch = preprocessor.batch_preprocess(image_paths, return_tensor=True)
        
        assert isinstance(batch, torch.Tensor)
        assert batch.shape[0] == 3
        assert batch.shape[1:] == (3, 224, 224)
    
    def test_real_world_inference(self):
        """Test RealWorldInference class."""
        from src.serving.inference import RealWorldInference, ImagePreprocessor
        from src.models.architectures import MiniCNN
        
        model = MiniCNN(in_channels=1, num_classes=10)
        preprocessor = ImagePreprocessor(target_size=28)
        
        inference = RealWorldInference(
            model=model,
            preprocessor=preprocessor,
            device="cpu"
        )
        
        # Create dummy image
        img = np.ones((28, 28, 3), dtype=np.uint8) * 128
        
        # Make prediction
        prediction = inference.predict(img, return_top_k=3)
        
        assert "predicted_class" in prediction
        assert "confidence" in prediction
        assert "top_k_predictions" in prediction
        assert len(prediction["top_k_predictions"]) <= 3
    
    def test_batch_inference(self):
        """Test batch inference."""
        from src.serving.inference import RealWorldInference, ImagePreprocessor
        from src.models.architectures import MiniCNN
        
        model = MiniCNN(in_channels=1, num_classes=10)
        preprocessor = ImagePreprocessor(target_size=28)
        
        inference = RealWorldInference(
            model=model,
            preprocessor=preprocessor,
            device="cpu"
        )
        
        # Create dummy images
        images = [
            np.ones((28, 28, 3), dtype=np.uint8) * 100,
            np.ones((32, 32, 3), dtype=np.uint8) * 150,
            np.ones((24, 24, 3), dtype=np.uint8) * 200
        ]
        
        # Batch predict
        predictions = inference.predict_batch(images, return_top_k=1)
        
        assert len(predictions) == 3
        for pred in predictions:
            assert "predicted_class" in pred
            assert "confidence" in pred
    
    def test_uncertainty_estimation(self):
        """Test prediction with uncertainty estimation."""
        from src.serving.inference import RealWorldInference, ImagePreprocessor
        from src.models.architectures import MiniCNN
        
        model = MiniCNN(in_channels=1, num_classes=10)
        preprocessor = ImagePreprocessor(target_size=28)
        
        inference = RealWorldInference(
            model=model,
            preprocessor=preprocessor,
            device="cpu"
        )
        
        # Create dummy image
        img = np.ones((28, 28, 3), dtype=np.uint8) * 128
        
        # Predict with uncertainty
        result = inference.predict_with_uncertainty(img, num_samples=5)
        
        assert "predicted_class" in result
        assert "mean_confidence" in result
        assert "confidence_std" in result
        assert "uncertainty" in result


class TestExplainability:
    """Test explainability utilities."""
    
    def test_gradcam_initialization(self):
        """Test Grad-CAM initialization."""
        from src.evaluation.explainability import GradCAM
        from src.models.architectures import MiniCNN
        
        model = MiniCNN(in_channels=1, num_classes=10)
        
        # GradCAM with conv layers
        gradcam = GradCAM(model, target_layer="conv_block.3", device="cpu")
        
        assert gradcam.model is not None
        assert gradcam.target_layer == "conv_block.3"
    
    def test_attention_mapper_init(self):
        """Test AttentionMapper initialization."""
        from src.evaluation.explainability import AttentionMapper
        from src.models.architectures import ResNet, BasicBlock
        
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        
        mapper = AttentionMapper(model, device="cpu")
        
        assert mapper.model is not None
        assert mapper.attention_maps is not None


class TestDataAugmentation:
    """Test data augmentation for inference."""
    
    def test_image_loader(self, tmp_path):
        """Test image loading utilities."""
        from src.serving.inference import load_and_preprocess_dataset, ImagePreprocessor
        
        # Create test images
        for i in range(3):
            img = Image.new('RGB', (256, 256), color='red')
            img_path = tmp_path / f"image_{i}.png"
            img.save(img_path)
        
        preprocessor = ImagePreprocessor(target_size=224)
        
        # Load dataset
        batch, paths = load_and_preprocess_dataset(tmp_path, preprocessor)
        
        assert isinstance(batch, torch.Tensor)
        assert len(paths) == 3
        assert batch.shape[0] == 3


class TestEndToEndInference:
    """End-to-end inference tests."""
    
    def test_model_to_inference_pipeline(self):
        """Test complete inference pipeline."""
        from src.models.architectures import MiniCNN
        from src.serving.inference import ImagePreprocessor, RealWorldInference
        
        # Create model
        model = MiniCNN(in_channels=1, num_classes=10)
        model.eval()
        
        # Create preprocessor
        preprocessor = ImagePreprocessor(target_size=28, normalize=False)
        
        # Create inference engine
        inference = RealWorldInference(
            model=model,
            preprocessor=preprocessor,
            device="cpu",
            confidence_threshold=0.3
        )
        
        # Create dummy image
        img_array = np.random.randint(0, 256, (28, 28, 3), dtype=np.uint8)
        
        # Inference
        result = inference.predict(img_array, return_top_k=5)
        
        # Verify result structure
        assert "predicted_class" in result
        assert "predicted_class_name" in result
        assert "confidence" in result
        assert "top_k_predictions" in result
        assert 0 <= result["predicted_class"] < 10
        assert 0 <= result["confidence"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
