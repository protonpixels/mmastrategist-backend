
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import uvicorn

from model import ufc_model

# ============================================
# PYDANTIC MODELS
# ============================================
class FighterMetrics(BaseModel):
    # Physical attributes
    height_cm: float = Field(180.0, ge=140, le=220, description="Height in centimeters")
    weight_in_kg: float = Field(77.0, ge=50, le=150, description="Weight in kilograms")

    # Striking metrics
    significant_strikes_landed_per_minute: float = Field(4.5, ge=0, le=15, description="Significant strikes landed per minute")
    significant_striking_accuracy: float = Field(55.0, ge=0, le=100, description="Significant striking accuracy percentage")
    significant_strikes_absorbed_per_minute: float = Field(3.0, ge=0, le=15, description="Significant strikes absorbed per minute")
    significant_strike_defence: float = Field(60.0, ge=0, le=100, description="Significant strike defence percentage")

    # Wrestling metrics
    average_takedowns_landed_per_15_minutes: float = Field(2.5, ge=0, le=10, description="Average takedowns landed per 15 minutes")
    takedown_accuracy: float = Field(50.0, ge=0, le=100, description="Takedown accuracy percentage")
    takedown_defense: float = Field(70.0, ge=0, le=100, description="Takedown defense percentage")

    # Grappling metrics
    average_submissions_attempted_per_15_minutes: float = Field(1.0, ge=0, le=10, description="Average submissions attempted per 15 minutes")

    # Stance
    stance: str = Field("Orthodox", description="Fighting stance")

class PredictionResponse(BaseModel):
    lose_probability: float
    lose_percentage: float
    confidence: str
    confidence_score: float  # Added for experience-based confidence
    tier: str
    color: str
    recommendation: str
    metrics: Dict[str, Any]

# ============================================
# FASTAPI APP
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("\n" + "="*60)
    print("UFC WIN PROBABILITY PREDICTOR API")
    print("="*60)

    if not ufc_model.load():
        print("\n⚠️ Model not found!")
        print("   Please run: python train_v2.py")
        print("   Then restart the server")
    else:
        print("\n✅ Server ready for predictions!")

    yield
    # Shutdown
    print("\n👋 Shutting down...")

app = FastAPI(
    title="UFC Win Probability Predictor",
    description="Predict win probability based on physical and technical metrics",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*","https://mmastar.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# ENDPOINTS
# ============================================
@app.get("/api/ufc/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": ufc_model.is_loaded,
        "features_count": len(ufc_model.get_feature_columns()) if ufc_model.is_loaded else 0
    }

@app.get("/api/ufc/defaults")
async def get_defaults():
    return {
        "height_cm": 180.0,
        "weight_in_kg": 77.0,
        "significant_strikes_landed_per_minute": 4.5,
        "significant_striking_accuracy": 55.0,
        "significant_strikes_absorbed_per_minute": 3.0,
        "significant_strike_defence": 60.0,
        "average_takedowns_landed_per_15_minutes": 2.5,
        "takedown_accuracy": 50.0,
        "takedown_defense": 70.0,
        "average_submissions_attempted_per_15_minutes": 1.0,
        "stance": "Orthodox"
    }

@app.get("/api/ufc/stances")
async def get_stances():
    return ["Orthodox", "Southpaw", "Switch", "Sideways"]

@app.post("/api/ufc/predict", response_model=PredictionResponse)
async def predict_win_probability(metrics: FighterMetrics):
    """Predict win probability based on fighter metrics"""
    try:
        if not ufc_model.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded. Please train first.")

        # Get prediction
        result = ufc_model.predict(metrics.dict())

        # Determine confidence and tier
        lose_prob = result['lose_probability']

        if lose_prob <= 20:
            confidence = "High"
            tier = "🏆 Elite Fighter"
            color = "#2ECC71"
            recommendation = "Exceptional metrics. You have a dominant skill set. Focus on maintaining these levels."
        elif lose_prob <= 40:
            confidence = "Good"
            tier = "⭐ Competitive Fighter"
            color = "#F39C12"
            recommendation = "Strong profile. Continue developing your strengths and work on weaknesses."
        elif lose_prob <= 75:
            confidence = "Moderate"
            tier = "⚖️ Developing Fighter"
            color = "#3498DB"
            recommendation = "Solid foundation. Focus on improving key areas identified below."
        else:
            confidence = "Low"
            tier = "🔴 Needs Improvement"
            color = "#E74C3C"
            recommendation = "Significant development needed. Focus on fundamentals first."

        # Calculate confidence score
        confidence_score = lose_prob

        return PredictionResponse(
            lose_probability=round(lose_prob, 2),
            lose_percentage=round(lose_prob, 2),
            confidence=confidence,
            confidence_score=confidence_score,
            tier=tier,
            color=color,
            recommendation=recommendation,
            metrics=metrics.dict(),
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ufc/docs")
async def get_docs():
    return {
        "description": "UFC Win Probability Predictor API",
        "version": "2.0.0",
        "features_used": [
            "height_cm",
            "weight_in_kg",
            "significant_strikes_landed_per_minute",
            "significant_striking_accuracy",
            "significant_strikes_absorbed_per_minute",
            "significant_strike_defence",
            "average_takedowns_landed_per_15_minutes",
            "takedown_accuracy",
            "takedown_defense",
            "average_submissions_attempted_per_15_minutes",
            "stance (one-hot encoded)"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)