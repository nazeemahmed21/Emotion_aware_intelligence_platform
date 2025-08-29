#!/usr/bin/env python3
"""
Professional installation script for the Emotion-Aware Voice Bot
"""
import subprocess
import sys
import os
import platform
import time
from pathlib import Path

class ProfessionalInstaller:
    """Professional installation manager"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = sys.version_info
        self.errors = []
        self.warnings = []
    
    def print_header(self):
        """Print installation header"""
        print("🎭" + "=" * 60 + "🎭")
        print("   EMOTION-AWARE VOICE BOT - PROFESSIONAL INSTALLATION")
        print("🎭" + "=" * 60 + "🎭")
        print()
        print(f"System: {platform.system()} {platform.release()}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"Architecture: {platform.machine()}")
        print()
    
    def check_python_version(self):
        """Check Python version compatibility"""
        print("🐍 Checking Python version...")
        
        if self.python_version < (3, 8):
            self.errors.append(f"Python 3.8+ required (current: {sys.version.split()[0]})")
            print(f"❌ Python 3.8+ required (current: {sys.version.split()[0]})")
            return False
        else:
            print(f"✅ Python {sys.version.split()[0]} is compatible")
            return True
    
    def check_virtual_environment(self):
        """Check if virtual environment is active"""
        print("\n🏠 Checking virtual environment...")
        
        in_venv = (hasattr(sys, 'real_prefix') or 
                  (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
        
        if in_venv:
            print("✅ Virtual environment is active")
            return True
        else:
            self.warnings.append("Virtual environment not detected")
            print("⚠️  Virtual environment not detected")
            print("   Recommendation: Create and activate a virtual environment")
            return False
    
    def install_core_dependencies(self):
        """Install core dependencies"""
        print("\n📦 Installing core dependencies...")
        
        core_packages = [
            "streamlit>=1.28.0",
            "numpy>=1.24.0", 
            "pandas>=2.0.0",
            "plotly>=5.15.0",
            "librosa>=0.10.0",
            "soundfile>=0.12.0",
            "openai-whisper>=20231117",
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "scikit-learn>=1.3.0",
            "requests>=2.31.0",
            "PyYAML>=6.0"
        ]
        
        for package in core_packages:
            try:
                print(f"  Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                print(f"  ✅ {package}")
            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to install {package}"
                self.errors.append(error_msg)
                print(f"  ❌ {package}")
        
        return len(self.errors) == 0
    
    def install_audio_dependencies(self):
        """Install audio recording dependencies (optional)"""
        print("\n🎙️ Installing audio recording dependencies...")
        
        audio_packages = [
            "streamlit-webrtc>=0.47.0",
            "audio-recorder-streamlit>=0.0.6"
        ]
        
        optional_packages = [
            "pyaudio"  # May fail on some systems
        ]
        
        # Install main audio packages
        for package in audio_packages:
            try:
                print(f"  Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                print(f"  ✅ {package}")
            except subprocess.CalledProcessError:
                self.warnings.append(f"Optional package {package} failed to install")
                print(f"  ⚠️  {package} (optional)")
        
        # Try optional packages
        for package in optional_packages:
            try:
                print(f"  Installing {package} (optional)...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                print(f"  ✅ {package}")
            except subprocess.CalledProcessError:
                self.warnings.append(f"Optional package {package} failed to install")
                print(f"  ⚠️  {package} (optional - may require system dependencies)")
    
    def check_system_dependencies(self):
        """Check system dependencies"""
        print("\n🔧 Checking system dependencies...")
        
        # Check FFmpeg
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
            print("✅ FFmpeg is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.warnings.append("FFmpeg not found")
            print("⚠️  FFmpeg not found")
            self._print_ffmpeg_instructions()
        
        # Check Ollama
        try:
            subprocess.run(['ollama', '--version'], 
                         capture_output=True, check=True)
            print("✅ Ollama is installed")
            self._check_ollama_models()
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.warnings.append("Ollama not found")
            print("⚠️  Ollama not found")
            self._print_ollama_instructions()
    
    def _print_ffmpeg_instructions(self):
        """Print FFmpeg installation instructions"""
        print("   FFmpeg installation:")
        if self.system == "windows":
            print("   • Download from https://ffmpeg.org/download.html")
            print("   • Or use: choco install ffmpeg")
            print("   • Or use: winget install FFmpeg")
        elif self.system == "darwin":
            print("   • Use: brew install ffmpeg")
        else:
            print("   • Use: sudo apt install ffmpeg (Ubuntu/Debian)")
            print("   • Use: sudo yum install ffmpeg (CentOS/RHEL)")
    
    def _print_ollama_instructions(self):
        """Print Ollama installation instructions"""
        print("   Ollama installation:")
        print("   • Download from https://ollama.ai/download")
        if self.system == "darwin":
            print("   • Or use: brew install ollama")
        elif self.system != "windows":
            print("   • Or use: curl -fsSL https://ollama.ai/install.sh | sh")
    
    def _check_ollama_models(self):
        """Check if Ollama models are available"""
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                models = result.stdout
                if 'llama2' in models or 'mistral' in models:
                    print("✅ Ollama models available")
                else:
                    self.warnings.append("No suitable Ollama models found")
                    print("⚠️  No suitable Ollama models found")
                    print("   Run: ollama pull llama2")
                    print("   Or:  ollama pull mistral")
        except Exception:
            pass
    
    def create_project_structure(self):
        """Create project directory structure"""
        print("\n📁 Creating project structure...")
        
        directories = [
            "data",
            "models", 
            "logs",
            "exports",
            ".streamlit"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            print(f"✅ Created {directory}/")
        
        # Create Streamlit config
        streamlit_config = """
[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
serverAddress = "localhost"
serverPort = 8501

[theme]
base = "light"
primaryColor = "#2E86AB"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
"""
        
        with open(".streamlit/config.toml", "w") as f:
            f.write(streamlit_config.strip())
        
        print("✅ Created Streamlit configuration")
    
    def run_tests(self):
        """Run installation tests"""
        print("\n🧪 Running installation tests...")
        
        # Test imports
        test_imports = [
            ("streamlit", "Streamlit"),
            ("numpy", "NumPy"),
            ("librosa", "Librosa"),
            ("whisper", "Whisper"),
            ("transformers", "Transformers"),
            ("torch", "PyTorch")
        ]
        
        for module, name in test_imports:
            try:
                __import__(module)
                print(f"✅ {name} import successful")
            except ImportError:
                self.errors.append(f"{name} import failed")
                print(f"❌ {name} import failed")
        
        # Test optional imports
        optional_imports = [
            ("streamlit_webrtc", "Streamlit WebRTC"),
            ("audio_recorder_streamlit", "Audio Recorder"),
            ("pyaudio", "PyAudio")
        ]
        
        for module, name in optional_imports:
            try:
                __import__(module)
                print(f"✅ {name} import successful (optional)")
            except ImportError:
                print(f"⚠️  {name} not available (optional)")
    
    def print_summary(self):
        """Print installation summary"""
        print("\n" + "=" * 60)
        print("📋 INSTALLATION SUMMARY")
        print("=" * 60)
        
        if not self.errors:
            print("🎉 Installation completed successfully!")
        else:
            print(f"❌ Installation completed with {len(self.errors)} errors:")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} warnings:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        print("\n🚀 Next steps:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull models: ollama pull llama2")
        print("3. Run the app: streamlit run professional_app.py")
        print("4. Or use: python run_professional.py")
        
        print("\n📖 Documentation:")
        print("• README.md - Setup and usage instructions")
        print("• professional_app.py - Main application")
        print("• src/ - Source code modules")
        
        return len(self.errors) == 0
    
    def install(self):
        """Run complete installation"""
        self.print_header()
        
        # Check prerequisites
        if not self.check_python_version():
            return False
        
        self.check_virtual_environment()
        
        # Install dependencies
        if not self.install_core_dependencies():
            print("❌ Core dependency installation failed")
            return False
        
        self.install_audio_dependencies()
        self.check_system_dependencies()
        self.create_project_structure()
        self.run_tests()
        
        return self.print_summary()

def main():
    """Main installation function"""
    installer = ProfessionalInstaller()
    
    try:
        success = installer.install()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Installation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()