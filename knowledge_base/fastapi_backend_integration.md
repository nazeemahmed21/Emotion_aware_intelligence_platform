6. FastAPI Backend Integration
Overview
FastAPI provides a high-performance backend for your emotion-aware voice bot, offering RESTful API endpoints, WebSocket support for real-time communication, and seamless integration with all your AI components.

Installation
bash
pip install fastapi uvicorn python-multipart python-jose[cryptography] passlib[bcrypt]
Basic FastAPI Application Structure
Main Application Setup
python
# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Optional
import uvicorn
import json
import asyncio
import tempfile
import os
from datetime import datetime
import numpy as np

# Import your processing modules
from whisper_transcription import transcribe_audio
from audio_processing import extract_audio_features
from emotion_detection import detect_emotion
from ollama_response import generate_empathetic_response

# Initialize FastAPI app
app = FastAPI(
    title="Emotion-Aware Voice Bot API",
    description="REST API for emotion detection from voice and empathetic response generation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use Redis or database)
conversation_history = []
active_connections = []
Data Models
Pydantic Models for Request/Response
python
# models.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class AudioInput(BaseModel):
    audio_data: str  # Base64 encoded audio
    sample_rate: int = 16000
    format: str = "wav"

class EmotionRequest(BaseModel):
    transcription: Optional[str] = None
    audio_data: Optional[str] = None  # Base64 encoded
    model: str = "default"

class EmotionResponse(BaseModel):
    transcription: str
    emotion: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    features: Dict[str, float]
    response: str
    timestamp: datetime

class Conversation(BaseModel):
    id: str
    user_input: str
    emotion: str
    confidence: float
    ai_response: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class AnalysisRequest(BaseModel):
    audio_file: UploadFile
    sensitivity: float = Field(0.7, ge=0.1, le=1.0)
    response_style: str = "empathetic"

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    models_loaded: Dict[str, bool]
REST API Endpoints
Health Check Endpoint
python
@app.get("/health", response_model=HealthCheck, tags=["Monitoring"])
async def health_check():
    """Check API health and model status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "models_loaded": {
            "whisper": True,
            "emotion_detection": True,
            "llm": True
        }
    }
Audio Processing Endpoints
python
@app.post("/api/analyze-audio", response_model=EmotionResponse, tags=["Analysis"])
async def analyze_audio_file(file: UploadFile = File(...)):
    """
    Analyze audio file for emotion and generate response
    """
    try:
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(400, "File must be an audio format")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Process audio pipeline
            transcription = transcribe_audio(tmp_path)
            features = extract_audio_features(tmp_path)
            emotion, confidence = detect_emotion(features)
            response = generate_empathetic_response(transcription, emotion, confidence)
            
            # Clean up
            os.unlink(tmp_path)
            
            return {
                "transcription": transcription,
                "emotion": emotion,
                "confidence": confidence,
                "features": features,
                "response": response,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            os.unlink(tmp_path)
            raise HTTPException(500, f"Processing error: {str(e)}")
            
    except Exception as e:
        raise HTTPException(500, f"Server error: {str(e)}")
Real-time Audio Analysis
python
@app.post("/api/analyze-audio-base64", response_model=EmotionResponse, tags=["Analysis"])
async def analyze_audio_base64(request: EmotionRequest):
    """
    Analyze base64 encoded audio data
    """
    try:
        import base64
        
        if not request.audio_data:
            raise HTTPException(400, "Audio data is required")
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Process audio
            transcription = transcribe_audio(tmp_path)
            features = extract_audio_features(tmp_path)
            emotion, confidence = detect_emotion(features)
            response = generate_empathetic_response(transcription, emotion, confidence)
            
            # Clean up
            os.unlink(tmp_path)
            
            # Store in conversation history
            conversation = {
                "id": str(len(conversation_history) + 1),
                "user_input": transcription,
                "emotion": emotion,
                "confidence": confidence,
                "ai_response": response,
                "timestamp": datetime.now()
            }
            conversation_history.append(conversation)
            
            return {
                "transcription": transcription,
                "emotion": emotion,
                "confidence": confidence,
                "features": features,
                "response": response,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            os.unlink(tmp_path)
            raise HTTPException(500, f"Processing error: {str(e)}")
            
    except Exception as e:
        raise HTTPException(500, f"Server error: {str(e)}")
Conversation Management Endpoints
python
@app.get("/api/conversations", response_model=List[Conversation], tags=["Conversations"])
async def get_conversations(limit: int = 10, offset: int = 0):
    """Get conversation history"""
    return conversation_history[offset:offset + limit]

@app.get("/api/conversations/{conversation_id}", response_model=Conversation, tags=["Conversations"])
async def get_conversation(conversation_id: str):
    """Get specific conversation by ID"""
    for conv in conversation_history:
        if conv["id"] == conversation_id:
            return conv
    raise HTTPException(404, "Conversation not found")

@app.delete("/api/conversations/{conversation_id}", tags=["Conversations"])
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    global conversation_history
    conversation_history = [conv for conv in conversation_history if conv["id"] != conversation_id]
    return {"message": "Conversation deleted"}

@app.delete("/api/conversations", tags=["Conversations"])
async def clear_conversations():
    """Clear all conversations"""
    global conversation_history
    conversation_history = []
    return {"message": "All conversations cleared"}
WebSocket Support for Real-time Communication
WebSocket Connection Manager
python
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()
WebSocket Endpoint for Real-time Audio
python
@app.websocket("/ws/audio-analysis")
async def websocket_audio_analysis(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio analysis
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive audio data (base64 encoded)
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "audio_chunk":
                # Process audio chunk
                audio_bytes = base64.b64decode(message["data"])
                
                # Save chunk to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name
                
                try:
                    # Process audio
                    transcription = transcribe_audio(tmp_path)
                    features = extract_audio_features(tmp_path)
                    emotion, confidence = detect_emotion(features)
                    
                    # Send intermediate results
                    await websocket.send_text(json.dumps({
                        "type": "intermediate_result",
                        "transcription": transcription,
                        "emotion": emotion,
                        "confidence": confidence,
                        "features": features
                    }))
                    
                    # Clean up
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    os.unlink(tmp_path)
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
            
            elif message["type"] == "finalize_analysis":
                # Generate final response
                response = generate_empathetic_response(
                    message["transcription"],
                    message["emotion"],
                    message["confidence"]
                )
                
                # Store conversation
                conversation = {
                    "id": str(len(conversation_history) + 1),
                    "user_input": message["transcription"],
                    "emotion": message["emotion"],
                    "confidence": message["confidence"],
                    "ai_response": response,
                    "timestamp": datetime.now()
                }
                conversation_history.append(conversation)
                
                # Send final response
                await websocket.send_text(json.dumps({
                    "type": "final_response",
                    "response": response,
                    "conversation_id": conversation["id"]
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"WebSocket error: {str(e)}"
        }))
        manager.disconnect(websocket)
Advanced Features
Batch Processing Endpoint
python
@app.post("/api/batch-analyze", tags=["Advanced"])
async def batch_analyze_audio(files: List[UploadFile] = File(...)):
    """
    Process multiple audio files in batch
    """
    results = []
    
    for file in files:
        try:
            # Process each file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            transcription = transcribe_audio(tmp_path)
            features = extract_audio_features(tmp_path)
            emotion, confidence = detect_emotion(features)
            
            results.append({
                "filename": file.filename,
                "transcription": transcription,
                "emotion": emotion,
                "confidence": confidence,
                "features": features
            })
            
            os.unlink(tmp_path)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}
Statistics and Analytics Endpoints
python
@app.get("/api/statistics/emotion-distribution", tags=["Statistics"])
async def get_emotion_distribution():
    """Get emotion distribution statistics"""
    emotions = [conv["emotion"] for conv in conversation_history]
    emotion_counts = {}
    
    for emotion in emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    return {
        "total_conversations": len(conversation_history),
        "emotion_distribution": emotion_counts,
        "average_confidence": np.mean([conv["confidence"] for conv in conversation_history]) if conversation_history else 0
    }

@app.get("/api/statistics/time-series", tags=["Statistics"])
async def get_emotion_time_series(hours: int = 24):
    """Get emotion time series data"""
    now = datetime.now()
    time_series = {}
    
    for conv in conversation_history:
        if (now - conv["timestamp"]).total_seconds() <= hours * 3600:
            hour_key = conv["timestamp"].strftime("%Y-%m-%d %H:00")
            if hour_key not in time_series:
                time_series[hour_key] = {}
            time_series[hour_key][conv["emotion"]] = time_series[hour_key].get(conv["emotion"], 0) + 1
    
    return time_series
Error Handling and Middleware
Custom Exception Handlers
python
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "success": False}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "success": False}
    )
Request Logging Middleware
python
from fastapi import Request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response
Configuration and Deployment
Configuration Management
python
# config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Emotion-Aware Voice Bot API"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    whisper_model: str = "base"
    emotion_model_path: str = "models/emotion_model.pkl"
    llm_model: str = "llama2"
    
    class Config:
        env_file = ".env"

settings = Settings()
Application Entry Point
python
# run.py
import uvicorn
from main import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload in development
        log_level="info"
    )
Docker Configuration
Dockerfile
dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
Docker Compose
yaml
# docker-compose.yml
version: '3.8'

services:
  emotion-bot-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=False
      - WHISPER_MODEL=base
    volumes:
      - ./models:/app/models
    restart: unless-stopped
API Documentation
FastAPI automatically generates interactive API documentation at:

Swagger UI: http://localhost:8000/docs

ReDoc: http://localhost:8000/redoc

This comprehensive FastAPI backend provides a robust, scalable foundation for your emotion-aware voice bot, with support for both RESTful APIs and real-time WebSocket communication.

