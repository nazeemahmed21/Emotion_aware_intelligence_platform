#!/usr/bin/env python3
"""
Clean installation for the minimal emotion analyzer
"""
import subprocess
import sys
import os

def print_header():
    print("🎭" + "=" * 40 + "🎭")
    print("   CLEAN EMOTION ANALYZER INSTALLATION")
    print("🎭" + "=" * 40 + "🎭")
    print()

def check_python():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required (current: {sys.version.split()[0]})")
        return False
    else:
        print(f"✅ Python {sys.version.split()[0]} is compatible")
        return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_clean.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def check_ollama():
    """Check Ollama installation"""
    print("\n🤖 Checking Ollama...")
    
    try:
        subprocess.run(['ollama', '--version'], capture_output=True, check=True)
        print("✅ Ollama is installed")
        
        # Check models
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            models = result.stdout
            if 'llama2' in models or 'mistral' in models:
                print("✅ Required models available")
            else:
                print("⚠️  No suitable models found")
                print("💡 Run: ollama pull llama2")
        
        return True
    except:
        print("⚠️  Ollama not found")
        print("💡 Install from: https://ollama.ai/download")
        return False

def test_imports():
    """Test if all imports work"""
    print("\n🧪 Testing imports...")
    
    test_modules = [
        'streamlit', 'numpy', 'pandas', 'plotly',
        'librosa', 'soundfile', 'whisper', 'torch', 'transformers'
    ]
    
    failed = []
    for module in test_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed.append(module)
    
    return len(failed) == 0

def main():
    print_header()
    
    # Check Python
    if not check_python():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Test imports
    if not test_imports():
        print("\n❌ Some imports failed. Try running:")
        print("pip install -r requirements_clean.txt")
        return False
    
    # Check Ollama
    check_ollama()
    
    print("\n" + "=" * 50)
    print("🎉 Installation completed!")
    print("\n🚀 Next steps:")
    print("1. Start Ollama: ollama serve")
    print("2. Pull models: ollama pull llama2")
    print("3. Run the app: python run_clean.py")
    print("\n📖 The app only supports file upload (no recording)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)