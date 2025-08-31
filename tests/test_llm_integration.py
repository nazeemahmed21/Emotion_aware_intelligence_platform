#!/usr/bin/env python3
"""
Test LLM Integration
====================

This script tests the LLM integration for the coaching system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_llm_imports():
    """Test if LLM modules can be imported"""
    try:
        from src.llm import LLMClient, LLMResponse
        print("✅ LLM imports successful")
        return True
    except ImportError as e:
        print(f"❌ LLM import failed: {e}")
        return False

def test_coaching_imports():
    """Test if coaching modules can be imported"""
    try:
        from src.coaching import CoachingAgent, CoachingContext, CoachingFeedback
        print("✅ Coaching imports successful")
        return True
    except ImportError as e:
        print(f"❌ Coaching import failed: {e}")
        return False

def test_config_loading():
    """Test if LLM configuration can be loaded"""
    try:
        from config import LLM_CONFIG
        print(f"✅ LLM config loaded: {LLM_CONFIG['provider']}")
        return True
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

def test_llm_client_creation():
    """Test if LLM client can be created"""
    try:
        from config import LLM_CONFIG
        from src.llm import LLMClient
        
        client = LLMClient(LLM_CONFIG)
        print(f"✅ LLM client created: {client.provider}")
        return client
    except Exception as e:
        print(f"❌ LLM client creation failed: {e}")
        return None

def test_ollama_connection():
    """Test Ollama connection"""
    try:
        from config import LLM_CONFIG
        from src.llm import LLMClient
        
        client = LLMClient(LLM_CONFIG)
        if client.provider == 'ollama':
            connection_ok = client.test_connection()
            if connection_ok:
                print("✅ Ollama connection successful")
            else:
                print("⚠️ Ollama connection failed - make sure Ollama is running")
            return connection_ok
        else:
            print(f"ℹ️ Not testing Ollama (provider: {client.provider})")
            return True
    except Exception as e:
        print(f"❌ Ollama connection test failed: {e}")
        return False

def test_coaching_agent():
    """Test coaching agent creation"""
    try:
        from src.coaching import CoachingAgent
        
        agent = CoachingAgent()
        print("✅ Coaching agent created successfully")
        return agent
    except Exception as e:
        print(f"❌ Coaching agent creation failed: {e}")
        return None

def main():
    """Run all tests"""
    print("🧠 Testing LLM Integration")
    print("=" * 40)
    
    # Test imports
    llm_ok = test_llm_imports()
    coaching_ok = test_coaching_imports()
    config_ok = test_config_loading()
    
    if not all([llm_ok, coaching_ok, config_ok]):
        print("\n❌ Basic imports failed - cannot proceed")
        return
    
    # Test LLM client
    client = test_llm_client_creation()
    if not client:
        print("\n❌ LLM client creation failed")
        return
    
    # Test connection
    connection_ok = test_ollama_connection()
    
    # Test coaching agent
    agent = test_coaching_agent()
    if not agent:
        print("\n❌ Coaching agent creation failed")
        return
    
    print("\n" + "=" * 40)
    if connection_ok:
        print("🎉 All tests passed! LLM integration is ready.")
        print("\n💡 To use AI coaching:")
        print("   1. Make sure Ollama is running")
        print("   2. Run: ollama run llama2")
        print("   3. Use the coaching feature in the Streamlit app")
    else:
        print("⚠️ Tests completed with warnings.")
        print("\n💡 The system will fall back to rule-based coaching")
        print("   until Ollama is available.")

if __name__ == "__main__":
    main()
