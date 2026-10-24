"""
FastAPI application for fraud detection service.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.predictor import load_predictor
from src.logger import setup_logger
from src.config import api_config, model_config

logger = setup_logger(__name__, "api.log")

# Initialize FastAPI app
app = FastAPI(
    title=api_config.title,
    description=api_config.description,
    version=api_config.version
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store loaded predictors in memory
predictors_cache = {}


# Pydantic models for request/response
class FraudTransaction(BaseModel):
    """Single fraud detection transaction."""
    user_id: Optional[int] = None
    purchase_value: float = Field(..., gt=0)
    age: int = Field(..., ge=18, le=120)
    hour_of_day: Optional[int] = Field(None, ge=0, lt=24)
    day_of_week: Optional[int] = Field(None, ge=0, lt=7)
    time_since_signup_hours: Optional[float] = Field(None, ge=0)
    # Add other features as needed
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": 12345,
                "purchase_value": 150.50,
                "age": 35,
                "hour_of_day": 14,
                "day_of_week": 2,
                "time_since_signup_hours": 48.5
            }
        }


class CreditCardTransaction(BaseModel):
    """Single credit card transaction."""
    Time: float
    Amount: float = Field(..., ge=0)
    # V1-V28 features
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    
    class Config:
        schema_extra = {
            "example": {
                "Time": 12345.0,
                "Amount": 250.00,
                "V1": -1.3598071336738,
                "V2": -0.0727811733098497,
                "V3": 2.53634673796914,
                # ... other V features
            }
        }


class PredictionRequest(BaseModel):
    """Request for batch predictions."""
    transactions: List[Dict[str, Any]]
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    include_details: bool = True


class PredictionResponse(BaseModel):
    """Response for prediction."""
    is_fraud: bool
    fraud_probability: float
    confidence: float
    risk_level: str
    threshold: float
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[PredictionResponse]
    total_transactions: int
    fraud_detected: int
    fraud_percentage: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    models_available: Dict[str, bool]


def get_predictor(model_type: str, dataset_type: str):
    """Get or load predictor from cache."""
    cache_key = f"{model_type}_{dataset_type}"
    
    if cache_key not in predictors_cache:
        try:
            logger.info(f"Loading predictor: {cache_key}")
            predictor = load_predictor(model_type, dataset_type)
            predictors_cache[cache_key] = predictor
            logger.info(f"Predictor loaded successfully: {cache_key}")
        except Exception as e:
            logger.error(f"Failed to load predictor {cache_key}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Model not available: {model_type} for {dataset_type}"
            )
    
    return predictors_cache[cache_key]


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Fraud Detection API",
        "version": api_config.version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    models_available = {}
    
    for model_type in model_config.model_types:
        for dataset_type in ['fraud', 'creditcard']:
            model_key = f"{model_type}_{dataset_type}"
            model_path = model_config.model_save_dir / f"{model_key}_model.joblib"
            models_available[model_key] = model_path.exists()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_available=models_available
    )


@app.post("/predict/fraud", response_model=PredictionResponse)
async def predict_fraud_single(
    transaction: FraudTransaction,
    model_type: str = "random_forest",
    threshold: float = 0.5
):
    """
    Predict fraud for a single e-commerce transaction.
    
    Args:
        transaction: Transaction data
        model_type: Model to use (logistic_regression, random_forest, xgboost)
        threshold: Classification threshold
    
    Returns:
        Prediction result with fraud probability and risk level
    """
    try:
        predictor = get_predictor(model_type, "fraud")
        
        # Convert to dict
        transaction_dict = transaction.dict()
        
        # Make prediction
        result = predictor.predict_single(transaction_dict, threshold=threshold)
        
        return PredictionResponse(
            is_fraud=result['is_fraud'],
            fraud_probability=result['fraud_probability'],
            confidence=result['confidence'],
            risk_level=predictor._get_risk_level(result['fraud_probability']),
            threshold=threshold,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/creditcard", response_model=PredictionResponse)
async def predict_creditcard_single(
    transaction: CreditCardTransaction,
    model_type: str = "random_forest",
    threshold: float = 0.5
):
    """
    Predict fraud for a single credit card transaction.
    
    Args:
        transaction: Transaction data
        model_type: Model to use
        threshold: Classification threshold
    
    Returns:
        Prediction result
    """
    try:
        predictor = get_predictor(model_type, "creditcard")
        
        # Convert to dict
        transaction_dict = transaction.dict()
        
        # Make prediction
        result = predictor.predict_single(transaction_dict, threshold=threshold)
        
        return PredictionResponse(
            is_fraud=result['is_fraud'],
            fraud_probability=result['fraud_probability'],
            confidence=result['confidence'],
            risk_level=predictor._get_risk_level(result['fraud_probability']),
            threshold=threshold,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/fraud/batch", response_model=BatchPredictionResponse)
async def predict_fraud_batch(
    request: PredictionRequest,
    model_type: str = "random_forest"
):
    """
    Predict fraud for multiple e-commerce transactions.
    
    Args:
        request: Batch prediction request with transactions
        model_type: Model to use
    
    Returns:
        Batch prediction results
    """
    try:
        predictor = get_predictor(model_type, "fraud")
        
        # Convert to DataFrame
        df = pd.DataFrame(request.transactions)
        
        # Make predictions
        results = predictor.predict_batch(
            df,
            threshold=request.threshold,
            include_details=request.include_details
        )
        
        # Format response
        predictions = []
        for idx, row in results.iterrows():
            predictions.append(PredictionResponse(
                is_fraud=bool(row['is_fraud']),
                fraud_probability=float(row['fraud_probability']),
                confidence=float(row.get('confidence', 0)),
                risk_level=row.get('risk_level', 'unknown'),
                threshold=request.threshold,
                timestamp=datetime.now().isoformat()
            ))
        
        fraud_count = results['is_fraud'].sum()
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_transactions=len(results),
            fraud_detected=int(fraud_count),
            fraud_percentage=float(fraud_count / len(results) * 100)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/creditcard/batch", response_model=BatchPredictionResponse)
async def predict_creditcard_batch(
    request: PredictionRequest,
    model_type: str = "random_forest"
):
    """
    Predict fraud for multiple credit card transactions.
    
    Args:
        request: Batch prediction request
        model_type: Model to use
    
    Returns:
        Batch prediction results
    """
    try:
        predictor = get_predictor(model_type, "creditcard")
        
        # Convert to DataFrame
        df = pd.DataFrame(request.transactions)
        
        # Make predictions
        results = predictor.predict_batch(
            df,
            threshold=request.threshold,
            include_details=request.include_details
        )
        
        # Format response
        predictions = []
        for idx, row in results.iterrows():
            predictions.append(PredictionResponse(
                is_fraud=bool(row['is_fraud']),
                fraud_probability=float(row['fraud_probability']),
                confidence=float(row.get('confidence', 0)),
                risk_level=row.get('risk_level', 'unknown'),
                threshold=request.threshold,
                timestamp=datetime.now().isoformat()
            ))
        
        fraud_count = results['is_fraud'].sum()
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_transactions=len(results),
            fraud_detected=int(fraud_count),
            fraud_percentage=float(fraud_count / len(results) * 100)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models."""
    available_models = {}
    
    for model_type in model_config.model_types:
        for dataset_type in ['fraud', 'creditcard']:
            model_key = f"{model_type}_{dataset_type}"
            model_path = model_config.model_save_dir / f"{model_key}_model.joblib"
            
            if model_path.exists():
                available_models[model_key] = {
                    "model_type": model_type,
                    "dataset": dataset_type,
                    "path": str(model_path),
                    "loaded": model_key in predictors_cache
                }
    
    return {
        "available_models": available_models,
        "total": len(available_models)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=api_config.host,
        port=api_config.port,
        reload=api_config.reload
    )
