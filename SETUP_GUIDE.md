# 🚀 Complete Setup Guide - Emotion-Aware Voice Intelligence Platform

This guide will help you set up both the React frontend and integrate it with your existing Python backend.

## 📋 Prerequisites

- **Node.js 16+** and npm/yarn
- **Python 3.8+** with your existing Hume AI setup
- **Modern web browser** with microphone access
- **Hume AI API key** (already configured in your backend)

## 🎯 Quick Start (5 minutes)

### 1. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create environment file
echo "REACT_APP_API_URL=http://localhost:5000" > .env

# Start development server
npm start
```

The React app will open at `http://localhost:3000`

### 2. Backend Integration (Choose One Option)

#### Option A: Quick Test with Mock Data
The frontend works immediately with mock data for testing the UI.

#### Option B: Flask API Server (Recommended)
```bash
# Install Flask (if not already installed)
pip install flask flask-cors

# Run the example API server
python backend_integration_example.py
```

#### Option C: Integrate with Your Existing Streamlit App
See the detailed integration steps below.

## 🔧 Detailed Integration Steps

### Frontend Configuration

1. **Environment Variables**
   ```bash
   # frontend/.env
   REACT_APP_API_URL=http://localhost:5000  # Your backend URL
   REACT_APP_ENVIRONMENT=development
   ```

2. **API Integration**
   The frontend expects a POST endpoint at `/api/analyze` that:
   - Accepts multipart/form-data with an audio file
   - Returns JSON with emotion analysis results

### Backend Integration Options

#### Option 1: Separate Flask API (Recommended)

Create a new Flask server that uses your existing Hume analysis code:

```python
# api_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os

# Import your existing functions
from emotion_aware_voice_analyzer import analyze_with_hume, extract_emotions

app = Flask(__name__)
CORS(app)

@app.route('/api/analyze', methods=['POST'])
def analyze_emotion():
    try:
        audio_file = request.files['audio']
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Use your existing Hume analysis
            predictions = analyze_with_hume(tmp_path)
            emotions = extract_emotions(predictions)
            
            # Transform for frontend (see backend_integration_example.py)
            response_data = transform_for_frontend(emotions)
            return jsonify(response_data)
            
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

#### Option 2: Modify Existing Streamlit App

Add API endpoints to your existing Streamlit app:

```python
# Add to emotion_aware_voice_analyzer.py
import streamlit as st
from streamlit.web import cli as stcli
import sys

# Add API endpoint handling
if st.query_params.get("api") == "analyze":
    # Handle API request
    handle_api_request()
else:
    # Your existing Streamlit UI code
    pass

def handle_api_request():
    # Process API request and return JSON
    pass
```

#### Option 3: FastAPI Integration

```python
# fastapi_server.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tempfile

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.post("/api/analyze")
async def analyze_emotion(audio: UploadFile = File(...)):
    # Your analysis logic here
    pass
```

### Data Format Requirements

The frontend expects this JSON response format:

```javascript
{
  "emotions": [
    {
      "name": "joy",
      "score": 0.85,
      "model": "prosody"
    }
    // ... more emotions
  ],
  "summary": {
    "totalEmotions": 10,
    "dominantEmotion": ["joy", {"meanScore": 0.85}],
    "averageIntensity": 0.433,
    "modelBreakdown": {
      "prosody": {"emotionCount": 4, "avgScore": 0.445},
      "burst": {"emotionCount": 3, "avgScore": 0.42},
      "language": {"emotionCount": 3, "avgScore": 0.437}
    },
    "hesitancyAnalysis": {
      "indicators": 0,
      "averageScore": 0,
      "level": "High",
      "patterns": []
    }
  },
  "aiResponse": "Generated analysis text..."
}
```

## 🎨 UI Features Overview

### Recording Interface
- **Large microphone button** - Primary recording control
- **Real-time waveform** - Visual feedback during recording
- **Timer display** - Shows recording duration
- **Pause/Resume** - Control recording without stopping
- **Audio playback** - Review before analysis

### Results Display
- **Emotion metrics** - Top emotions with confidence scores
- **Visual charts** - Progress bars and rankings
- **AI insights** - Generated analysis text
- **Model breakdown** - Shows which AI model detected each emotion
- **Hesitancy analysis** - Specialized uncertainty detection

### Responsive Design
- **Desktop** - Full-width layout with side-by-side components
- **Tablet** - Stacked layout with optimized spacing
- **Mobile** - Single-column layout with touch-friendly controls

## 🔒 Security Considerations

### Audio Privacy
- Audio files are processed temporarily and not stored
- Files are automatically deleted after analysis
- No audio data is sent to external services (except Hume AI)

### CORS Configuration
```python
# For Flask
from flask_cors import CORS
CORS(app, origins=["http://localhost:3000"])  # Development
CORS(app, origins=["https://yourdomain.com"])  # Production
```

### Environment Variables
```bash
# Never commit these to version control
HUME_API_KEY=your_key_here
HUME_SECRET_KEY=your_secret_here
```

## 🚀 Production Deployment

### Frontend Deployment

1. **Build for production**
   ```bash
   cd frontend
   npm run build
   ```

2. **Deploy to hosting service**
   - **Netlify**: Drag and drop `build` folder
   - **Vercel**: Connect Git repository
   - **AWS S3**: Upload build files

3. **Update environment variables**
   ```bash
   REACT_APP_API_URL=https://your-api-domain.com
   ```

### Backend Deployment

1. **Containerize with Docker**
   ```dockerfile
   FROM python:3.9
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 5000
   CMD ["python", "api_server.py"]
   ```

2. **Deploy to cloud service**
   - **Heroku**: `git push heroku main`
   - **AWS ECS**: Deploy container
   - **Google Cloud Run**: Deploy container

## 🛠️ Development Workflow

### 1. Frontend Development
```bash
cd frontend
npm start  # Runs on localhost:3000
```

### 2. Backend Development
```bash
python backend_integration_example.py  # Runs on localhost:5000
```

### 3. Testing Integration
1. Start both frontend and backend
2. Open `http://localhost:3000`
3. Record audio and test analysis
4. Check browser console for any errors

## 📊 Performance Optimization

### Frontend
- **Code splitting** - Automatic with React
- **Image optimization** - WebP format with fallbacks
- **Bundle analysis** - `npm run analyze`

### Backend
- **Async processing** - Use async/await for Hume API calls
- **Caching** - Cache common analysis results
- **Rate limiting** - Prevent API abuse

## 🐛 Troubleshooting

### Common Issues

1. **CORS Errors**
   ```
   Solution: Add CORS headers to your backend
   ```

2. **Microphone Access Denied**
   ```
   Solution: Use HTTPS in production, check browser permissions
   ```

3. **Audio Format Issues**
   ```
   Solution: Frontend uses WebM format, ensure backend can handle it
   ```

4. **API Connection Failed**
   ```
   Solution: Check REACT_APP_API_URL in .env file
   ```

### Debug Mode

Enable debug logging:
```javascript
// In frontend/src/api/emotionApi.js
console.log('Sending audio blob:', audioBlob);
console.log('API response:', results);
```

## 📞 Support

### Getting Help
1. Check the browser console for errors
2. Verify API endpoint is responding: `curl http://localhost:5000/health`
3. Test with mock data first
4. Check network tab in browser dev tools

### Next Steps
1. Test the basic setup with mock data
2. Integrate with your existing Hume analysis code
3. Customize the UI to match your brand
4. Deploy to production

---

**🎉 You're ready to build amazing voice emotion analysis experiences!**