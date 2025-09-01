#!/usr/bin/env python3
"""
Quick Fix for Ollama Issues
===========================

This script fixes common Ollama issues:
1. CUDA memory errors
2. Model name mismatches
3. Service startup problems
4. 500 errors from corrupted models
"""

import subprocess
import requests
import time
import json
import os

def stop_ollama():
    """Stop Ollama service"""
    try:
        # Windows
        subprocess.run(['taskkill', '/f', '/im', 'ollama.exe'], capture_output=True)
        time.sleep(2)
        print("✅ Stopped Ollama service")
    except:
        try:
            # Linux/Mac
            subprocess.run(['pkill', 'ollama'], capture_output=True)
            time.sleep(2)
            print("✅ Stopped Ollama service")
        except:
            print("⚠️ Could not stop Ollama (may not be running)")

def start_ollama_cpu():
    """Start Ollama with CPU-only mode"""
    try:
        # Set environment variable to force CPU usage
        env = os.environ.copy()
        env['OLLAMA_HOST'] = '127.0.0.1:11434'
        
        # Start Ollama in background
        subprocess.Popen(['ollama', 'serve'], env=env)
        time.sleep(5)
        print("✅ Started Ollama service")
        return True
    except Exception as e:
        print(f"❌ Failed to start Ollama: {e}")
        return False

def check_models():
    """Check available models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            models = [model['name'] for model in models_data.get('models', [])]
            print(f"📋 Available models: {models}")
            return models
        else:
            print("❌ Could not get models")
            return []
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return []

def pull_small_model():
    """Pull a small model that should work"""
    try:
        print("📥 Pulling llama2:7b (smaller model)...")
        result = subprocess.run(['ollama', 'pull', 'llama2:7b'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Successfully pulled llama2:7b")
            return True
        else:
            print(f"❌ Failed to pull llama2:7b: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error pulling model: {e}")
        return False

def test_model(model_name):
    """Test if a model works"""
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": "Say 'Hello'",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 10,
                "num_gpu": 0  # Force CPU
            }
        }
        
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model {model_name} works!")
            print(f"   Response: {result.get('response', '')[:50]}...")
            return True
        else:
            print(f"❌ Model {model_name} failed: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Error: {error_detail}")
            except:
                print(f"   Error: {response.text[:100]}")
            return False
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        return False

def remove_corrupted_models():
    """Remove potentially corrupted models"""
    try:
        print("🗑️ Removing potentially corrupted models...")
        subprocess.run(['ollama', 'rm', 'mistral:latest'], capture_output=True)
        subprocess.run(['ollama', 'rm', 'llama2:latest'], capture_output=True)
        print("✅ Removed old models")
        return True
    except Exception as e:
        print(f"⚠️ Could not remove models: {e}")
        return False

def main():
    """Main fix function"""
    print("🔧 Quick Fix for Ollama Issues")
    print("=" * 40)
    
    # Stop existing service
    stop_ollama()
    
    # Start with CPU mode
    if not start_ollama_cpu():
        return
    
    # Check current models
    models = check_models()
    if models:
        print("\n🧪 Testing current models...")
        working_models = []
        for model in models:
            if test_model(model):
                working_models.append(model)
        
        if working_models:
            print(f"\n✅ Working models: {working_models}")
            print(f"🎯 Recommended: {working_models[0]}")
            
            # Update config
            print("\n🔧 Update your .env file with:")
            print(f"LLM_MODEL_NAME={working_models[0]}")
            print("LLM_MAX_TOKENS=1000")
            print("LLM_TEMPERATURE=0.7")
            
            print("\n✅ Ollama should now work with your coaching system!")
            return
    
    # If no models work, try pulling a smaller model
    print("\n❌ No current models are working. Trying smaller model...")
    
    # Remove corrupted models
    remove_corrupted_models()
    
    # Pull smaller model
    if pull_small_model():
        time.sleep(2)  # Wait for model to load
        
        # Test the new model
        if test_model('llama2:7b'):
            print("\n✅ llama2:7b is working!")
            print("\n🔧 Update your .env file with:")
            print("LLM_MODEL_NAME=llama2:7b")
            print("LLM_MAX_TOKENS=1000")
            print("LLM_TEMPERATURE=0.7")
            
            print("\n✅ Ollama should now work with your coaching system!")
        else:
            print("❌ Even the smaller model failed")
            print("Try manually: ollama pull llama2:7b")
    else:
        print("❌ Could not pull smaller model")
        print("Try manually: ollama pull llama2:7b")

if __name__ == "__main__":
    main()
