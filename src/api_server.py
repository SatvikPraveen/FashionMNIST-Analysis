"""
FastAPI server for FashionMNIST-Analysis.

REST API for model inference with real-time predictions and model comparison.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import logging
from typing import List, Optional
from pydantic import BaseModel
import io
from PIL import Image
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FashionMNIST API",
    description="REST API for Fashion MNIST classification with transfer learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
class ModelState:
    """Global model state."""
    model = None
    preprocessor = None
    inference_engine = None
    ensemble = None
    config = None

model_state = ModelState()


# Request/Response models
class PredictionResponse(BaseModel):
    """Prediction response model."""
    predicted_class: int
    predicted_class_name: str
    confidence: float
    top_k_predictions: List[dict]
    all_probabilities: dict


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    num_samples: int


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    num_parameters: int
    num_classes: int
    input_size: int
    device: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    message: str


# Utility functions
def get_num_parameters(model: torch.nn.Module) -> int:
    """Get total number of model parameters."""
    return sum(p.numel() for p in model.parameters())


# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        Health status and model state
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model_state.model is not None,
        message="API is operational"
    )


@app.post("/initialize")
async def initialize_model(
    model_path: str = Query(..., description="Path to model weights"),
    config_path: str = Query(..., description="Path to configuration file"),
    transfer_learning: bool = Query(False, description="Use transfer learning model")
) -> JSONResponse:
    """
    Initialize model and inference engine.
    
    Args:
        model_path: Path to saved model weights
        config_path: Path to configuration file
        transfer_learning: Whether to use transfer learning model
        
    Returns:
        Model information
    """
    try:
        from src.config import load_config
        from src.real_world_inference import ImagePreprocessor, RealWorldInference
        from src.model_definitions import ResNet, BasicBlock
        from src.transfer_learning import TransferLearningModel
        
        # Load config
        model_state.config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")
        
        # Load model
        device = model_state.config.model.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if transfer_learning:
            tl_model = TransferLearningModel(
                model_name=model_state.config.model.architecture,
                num_classes=model_state.config.model.num_classes,
                pretrained=model_state.config.model.pretrained,
                device=device
            )
            model_state.model = tl_model.model
        else:
            model_state.model = ResNet(BasicBlock, [2, 2, 2, 2], 
                                       num_classes=model_state.config.model.num_classes)
            model_state.model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True)
            )
            model_state.model = model_state.model.to(device).eval()
        
        # Initialize preprocessor and inference engine
        model_state.preprocessor = ImagePreprocessor(
            target_size=model_state.config.data.image_size,
            normalize=model_state.config.data.normalize,
            device=device
        )
        
        model_state.inference_engine = RealWorldInference(
            model_state.model,
            model_state.preprocessor,
            device=device,
            confidence_threshold=model_state.config.inference.confidence_threshold
        )
        
        logger.info("Model initialized successfully")
        
        return JSONResponse({
            "status": "success",
            "message": "Model initialized successfully",
            "model_info": {
                "model_name": model_state.config.model.architecture,
                "num_parameters": get_num_parameters(model_state.model),
                "num_classes": model_state.config.model.num_classes,
                "device": device
            }
        })
    
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info() -> ModelInfo:
    """Get model information."""
    if model_state.model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    
    device = next(model_state.model.parameters()).device
    
    return ModelInfo(
        model_name=model_state.config.model.architecture if model_state.config else "unknown",
        num_parameters=get_num_parameters(model_state.model),
        num_classes=model_state.config.model.num_classes if model_state.config else 10,
        input_size=model_state.config.data.image_size if model_state.config else 224,
        device=str(device)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), top_k: int = Query(3)) -> PredictionResponse:
    """
    Make prediction on uploaded image.
    
    Args:
        file: Uploaded image file
        top_k: Return top K predictions
        
    Returns:
        Prediction with confidence scores
    """
    if model_state.inference_engine is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_array = np.array(image)
        
        # Make prediction
        prediction = model_state.inference_engine.predict(image_array, return_top_k=top_k)
        
        return PredictionResponse(
            predicted_class=prediction["predicted_class"],
            predicted_class_name=prediction["predicted_class_name"],
            confidence=prediction["confidence"],
            top_k_predictions=prediction["top_k_predictions"],
            all_probabilities=prediction["all_probabilities"]
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: List[UploadFile] = File(...),
    top_k: int = Query(1)
) -> BatchPredictionResponse:
    """
    Make predictions on multiple images.
    
    Args:
        files: List of uploaded image files
        top_k: Return top K predictions
        
    Returns:
        Batch predictions
    """
    if model_state.inference_engine is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    
    try:
        # Read images
        images = []
        for file in files:
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            images.append(np.array(image))
        
        # Make predictions
        predictions_list = model_state.inference_engine.predict_batch(images, return_top_k=top_k)
        
        # Format responses
        responses = [
            PredictionResponse(
                predicted_class=pred["predicted_class"],
                predicted_class_name=pred["top_k_predictions"][0]["class_name"],
                confidence=pred["top_k_predictions"][0]["confidence"],
                top_k_predictions=pred["top_k_predictions"],
                all_probabilities={}
            )
            for pred in predictions_list
        ]
        
        return BatchPredictionResponse(
            predictions=responses,
            num_samples=len(responses)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/uncertainty")
async def predict_with_uncertainty(file: UploadFile = File(...), num_samples: int = Query(10)):
    """
    Make prediction with uncertainty estimation.
    
    Uses Monte Carlo dropout for uncertainty quantification.
    
    Args:
        file: Uploaded image file
        num_samples: Number of stochastic forward passes
        
    Returns:
        Prediction with uncertainty estimates
    """
    if model_state.inference_engine is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_array = np.array(image)
        
        # Make prediction with uncertainty
        result = model_state.inference_engine.predict_with_uncertainty(
            image_array, num_samples=num_samples
        )
        
        return JSONResponse(result)
    
    except Exception as e:
        logger.error(f"Uncertainty prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_status() -> JSONResponse:
    """Get API and model status."""
    return JSONResponse({
        "api_status": "operational",
        "model_initialized": model_state.model is not None,
        "config_loaded": model_state.config is not None,
        "device": str(next(model_state.model.parameters()).device) if model_state.model else "none"
    })


if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(level=logging.INFO)
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
