# 🧠 LLM Integration Setup Guide

This guide will help you set up the LLM integration for AI-powered coaching feedback.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Choose Your LLM Provider

#### Option A: Ollama (Local - Recommended for Development)
```bash
# Install Ollama from https://ollama.ai/
# Then run:
ollama run llama2
```

#### Option B: OpenAI
```bash
# Set environment variable:
export OPENAI_API_KEY="your-api-key-here"
```

#### Option C: Anthropic Claude
```bash
# Set environment variable:
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 3. Configure Environment Variables

Create a `.env` file in your project root:

```env
# LLM Configuration
LLM_PROVIDER=ollama  # or 'openai' or 'anthropic'
LLM_MODEL_NAME=llama2  # or 'gpt-4' or 'claude-3-sonnet'
LLM_BASE_URL=http://localhost:11434  # for Ollama
LLM_API_KEY=your-api-key  # for OpenAI/Anthropic
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
LLM_TIMEOUT_SECONDS=60
LLM_MAX_RETRIES=3

# Hume AI (Required)
HUME_API_KEY=your-hume-api-key
```

### 4. Test the Integration

```bash
python tests/test_llm_integration.py
```

## 🔧 Configuration Details

### LLM Provider Options

| Provider | Model Examples | Setup Complexity | Cost |
|----------|----------------|------------------|------|
| **Ollama** | llama2, mistral, codellama | Low | Free |
| **OpenAI** | gpt-4, gpt-3.5-turbo | Medium | Pay-per-use |
| **Anthropic** | claude-3-sonnet, claude-3-haiku | Medium | Pay-per-use |

### Model Recommendations

#### For Interview Coaching:
- **Ollama**: `llama2:13b` or `mistral:7b`
- **OpenAI**: `gpt-4` or `gpt-3.5-turbo`
- **Anthropic**: `claude-3-sonnet`

#### For Technical Questions:
- **Ollama**: `codellama:13b` or `llama2:13b`
- **OpenAI**: `gpt-4` or `gpt-3.5-turbo`
- **Anthropic**: `claude-3-sonnet`

## 🧪 Testing

### Test LLM Connection
```python
from src.llm import LLMClient
from config import LLM_CONFIG

client = LLMClient(LLM_CONFIG)
if client.test_connection():
    print("✅ LLM connection successful!")
else:
    print("❌ LLM connection failed")
```

### Test Coaching Agent
```python
from src.coaching import CoachingAgent

agent = CoachingAgent()
print("✅ Coaching agent ready!")
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Ollama Connection Failed
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve

# In another terminal, pull a model
ollama pull llama2
```

#### 2. Import Errors
```bash
# Make sure you're in the right directory
cd Emotion_aware_intelligence_platform

# Install dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 3. Model Not Found
```bash
# For Ollama, check available models
ollama list

# Pull a specific model
ollama pull llama2:7b
```

### Performance Tips

1. **Use smaller models for faster responses**: `llama2:7b` instead of `llama2:13b`
2. **Adjust temperature**: Lower values (0.1-0.3) for consistent responses, higher (0.7-0.9) for creativity
3. **Limit max tokens**: 1000-2000 for coaching feedback is usually sufficient

## 🎯 What Happens When LLM is Unavailable?

The system gracefully falls back to rule-based coaching:

1. **AI Analysis**: Attempts to use configured LLM
2. **Fallback**: If LLM fails, uses pre-built rules and analysis
3. **User Experience**: Seamless transition with clear status indicators

## 🔮 Future Enhancements

- **Multi-model routing**: Automatically switch between models based on task
- **Caching**: Store common coaching responses for faster feedback
- **Fine-tuning**: Custom models trained on interview coaching data
- **Batch processing**: Analyze multiple answers simultaneously

## 📞 Support

If you encounter issues:

1. Check the test script: `python tests/test_llm_integration.py`
2. Review environment variables and configuration
3. Check LLM provider status (Ollama, OpenAI, etc.)
4. Review logs for detailed error messages

---

**Happy Coaching! 🎉**
