#!/usr/bin/env python3
"""
Ollama Setup Script for Emotion-Aware Intelligence Platform
===========================================================

This script helps set up and configure Ollama for the coaching system.
"""

import requests
import json
import subprocess
import sys
import time
from pathlib import Path

def check_ollama_installation():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama is not properly installed")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        return False

def check_ollama_service():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            return True
        else:
            print(f"❌ Ollama service returned HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama service")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama service: {e}")
        return False

def get_available_models():
    """Get list of available models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            models = [model['name'] for model in models_data.get('models', [])]
            return models
        else:
            return []
    except Exception as e:
        print(f"Error getting models: {e}")
        return []

def pull_model(model_name):
    """Pull a specific model"""
    print(f"📥 Pulling model: {model_name}")
    try:
        result = subprocess.run(['ollama', 'pull', model_name], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Successfully pulled {model_name}")
            return True
        else:
            print(f"❌ Failed to pull {model_name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error pulling model: {e}")
        return False

def test_model_generation(model_name):
    """Test if a model can generate responses"""
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": "Hello, this is a test. Please respond with 'Test successful'.",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 50
            }
        }
        
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model {model_name} test successful")
            print(f"   Response: {result.get('response', '')[:100]}...")
            return True
        else:
            print(f"❌ Model {model_name} test failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing model {model_name}: {e}")
        return False

def main():
    """Main setup function"""
    print("🔧 Ollama Setup for Emotion-Aware Intelligence Platform")
    print("=" * 60)
    
    # Check installation
    if not check_ollama_installation():
        print("\n📥 To install Ollama:")
        print("   Visit: https://ollama.ai/download")
        print("   Or run: curl -fsSL https://ollama.ai/install.sh | sh")
        return
    
    # Check service
    if not check_ollama_service():
        print("\n🚀 Starting Ollama service...")
        try:
            subprocess.Popen(['ollama', 'serve'])
            time.sleep(3)  # Wait for service to start
            
            if not check_ollama_service():
                print("❌ Failed to start Ollama service")
                return
        except Exception as e:
            print(f"❌ Error starting Ollama: {e}")
            return
    
    # Get available models
    print("\n📋 Available models:")
    models = get_available_models()
    if models:
        for model in models:
            print(f"   - {model}")
    else:
        print("   No models found")
    
    # Check if required models are available
    required_models = ['mistral', 'llama2', 'codellama']
    missing_models = [model for model in required_models if model not in models]
    
    if missing_models:
        print(f"\n📥 Missing models: {', '.join(missing_models)}")
        print("Pulling recommended models...")
        
        for model in missing_models:
            if pull_model(model):
                models.append(model)
            else:
                print(f"⚠️ Failed to pull {model}, continuing with available models")
    
    # Test model generation
    print("\n🧪 Testing model generation...")
    test_models = ['mistral', 'llama2'] if 'mistral' in models or 'llama2' in models else models[:1]
    
    for model in test_models:
        if test_model_generation(model):
            print(f"✅ {model} is ready for use")
            break
    else:
        print("❌ No models are working properly")
        return
    
    # Configuration summary
    print("\n📊 Configuration Summary:")
    print(f"   Ollama URL: http://localhost:11434")
    print(f"   Available models: {', '.join(models)}")
    print(f"   Recommended model: mistral")
    
    # Environment variables
    print("\n🔧 Environment Variables (add to your .env file):")
    print("   LLM_PROVIDER=ollama")
    print("   LLM_MODEL_NAME=mistral")
    print("   LLM_BASE_URL=http://localhost:11434")
    print("   LLM_TEMPERATURE=0.7")
    print("   LLM_MAX_TOKENS=1000")
    
    print("\n✅ Ollama setup complete!")
    print("You can now use the AI coaching features in your application.")

if __name__ == "__main__":
    main()
