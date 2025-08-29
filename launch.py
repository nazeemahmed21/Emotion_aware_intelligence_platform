#!/usr/bin/env python3
"""
Professional Launcher for Emotion-Aware Voice Intelligence Platform
=====================================================================

This launcher provides a clean, professional way to start the application with:
- Comprehensive dependency checking
- Environment validation
- Proper error handling and user feedback
- Production-ready configuration

Author: Emotion AI Team
Version: 2.0.0
License: MIT
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ApplicationLauncher:
    """Professional application launcher with comprehensive validation"""
    
    def __init__(self):
        self.app_name = "Emotion-Aware Voice Intelligence Platform"
        self.main_file = "emotion_aware_voice_analyzer.py"
        self.requirements_file = "requirements.txt"
        self.env_file = ".env"
        
    def print_header(self) -> None:
        """Print professional application header"""
        print("\n" + "=" * 70)
        print(f"🧠 {self.app_name}")
        print("Enterprise-grade emotion analysis powered by Hume AI")
        print("=" * 70)
        
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements"""
        min_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version >= min_version:
            print(f"✅ Python {'.'.join(map(str, current_version))} (meets requirement >= {'.'.join(map(str, min_version))})")
            return True
        else:
            print(f"❌ Python {'.'.join(map(str, current_version))} is too old (requires >= {'.'.join(map(str, min_version))})")
            return False
    
    def check_required_files(self) -> bool:
        """Check if all required files exist"""
        required_files = [self.main_file, self.requirements_file]
        missing_files = []
        
        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing required files: {', '.join(missing_files)}")
            return False
        
        print("✅ All required files are present")
        return True
    
    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """Check if all required packages are installed"""
        required_packages = [
            ('streamlit', 'streamlit'),
            ('numpy', 'numpy'),
            ('pandas', 'pandas'),
            ('plotly', 'plotly'),
            ('dotenv', 'python-dotenv'),
            ('audio_recorder_streamlit', 'audio-recorder-streamlit'),
        ]
        
        missing_packages = []
        
        for package_name, pip_name in required_packages:
            try:
                __import__(package_name)
            except ImportError:
                missing_packages.append(pip_name)
        
        if missing_packages:
            print(f"❌ Missing required packages: {', '.join(missing_packages)}")
            print(f"   Install with: pip install {' '.join(missing_packages)}")
            return False, missing_packages
        
        print("✅ All required packages are installed")
        return True, []
    
    def check_environment(self) -> bool:
        """Check if environment is properly configured"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            print("❌ python-dotenv not installed")
            return False
        
        # Check for .env file
        if not Path(self.env_file).exists():
            print(f"⚠️  {self.env_file} file not found")
            print("   Create one from .env.example and add your Hume AI API key")
            return False
        
        # Check required environment variables
        api_key = os.getenv('HUME_API_KEY')
        if not api_key:
            print("❌ HUME_API_KEY not found in environment")
            print(f"   Please add your Hume AI API key to the {self.env_file} file")
            return False
        
        # Validate API key format (basic check)
        if len(api_key) < 10:
            print("❌ HUME_API_KEY appears to be invalid (too short)")
            return False
        
        print(f"✅ Environment configuration is valid (API key: {api_key[:10]}...)")
        return True
    
    def check_hume_integration(self) -> bool:
        """Check if Hume AI integration is available"""
        try:
            knowledge_base_path = Path("knowledge_base")
            if not knowledge_base_path.exists():
                print("⚠️  knowledge_base directory not found")
                print("   Hume AI integration may not work properly")
                return False
            
            hume_path = knowledge_base_path / "hume"
            if not hume_path.exists():
                print("⚠️  Hume AI integration files not found")
                return False
            
            print("✅ Hume AI integration is available")
            return True
            
        except Exception as e:
            print(f"⚠️  Could not verify Hume AI integration: {e}")
            return False
    
    def run_pre_flight_checks(self) -> bool:
        """Run all pre-flight checks"""
        print("\n🔍 Running pre-flight checks...")
        print("-" * 40)
        
        checks = [
            ("Python Version", self.check_python_version),
            ("Required Files", self.check_required_files),
            ("Dependencies", lambda: self.check_dependencies()[0]),
            ("Environment", self.check_environment),
            ("Hume Integration", self.check_hume_integration),
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            try:
                if not check_func():
                    all_passed = False
            except Exception as e:
                print(f"❌ {check_name} check failed: {e}")
                all_passed = False
        
        print("-" * 40)
        if all_passed:
            print("✅ All pre-flight checks passed!")
        else:
            print("❌ Some pre-flight checks failed. Please fix the issues above.")
        
        return all_passed
    
    def launch_application(self) -> None:
        """Launch the Streamlit application"""
        print("\n🚀 Starting the application...")
        print("📱 The app will open in your default browser")
        print("🔗 URL: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the application")
        print("=" * 70)
        
        try:
            # Streamlit configuration for production
            cmd = [
                sys.executable, "-m", "streamlit", "run", 
                self.main_file,
                "--server.headless", "false",
                "--server.runOnSave", "true",
                "--browser.gatherUsageStats", "false",
                "--server.maxUploadSize", "200",
                "--server.enableCORS", "false",
                "--server.enableXsrfProtection", "true"
            ]
            
            logger.info(f"Launching application with command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
        except KeyboardInterrupt:
            print("\n👋 Application stopped by user")
            logger.info("Application stopped by user interrupt")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error starting application: {e}")
            logger.error(f"Subprocess error: {e}")
            sys.exit(1)
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            logger.error(f"Unexpected error during launch: {e}")
            sys.exit(1)
    
    def run(self) -> None:
        """Main launcher execution"""
        self.print_header()
        
        # Run pre-flight checks
        if not self.run_pre_flight_checks():
            print("\n❌ Cannot start application due to failed checks.")
            print("Please resolve the issues above and try again.")
            sys.exit(1)
        
        # Launch application
        self.launch_application()

def main():
    """Main entry point"""
    launcher = ApplicationLauncher()
    launcher.run()

if __name__ == "__main__":
    main()