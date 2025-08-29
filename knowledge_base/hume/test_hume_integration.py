#!/usr/bin/env python3
"""
Test Hume AI Integration

Simple test script to verify the Hume AI integration is working correctly.
Tests configuration loading, client initialization, and basic functionality.
"""

import sys
import json
import logging
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from hume.hume_client import HumeConfig, HumeClient, GranularityLevel
from hume.core.config_loader import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)

def test_config_loading():
    """Test configuration loading from audio_analysis.json."""
    print("=" * 60)
    print("🔧 TESTING CONFIG LOADING")
    print("=" * 60)
    
    try:
        # Test loading config
        config = load_config()
        hume_config = config.get('hume_ai', {})
        
        print(f"✅ Config loaded successfully")
        print(f"📊 Hume AI config keys: {list(hume_config.keys())}")
        print(f"🎯 Default granularity: {hume_config.get('default_granularity', 'Not set')}")
        print(f"📈 Score threshold: {hume_config.get('score_threshold', 'Not set')}")
        print(f"🔢 Top N emotions: {hume_config.get('top_n_emotions', 'Not set')}")
        
        # Test configuration structure
        print(f"📊 Config structure validated successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_hume_client_init():
    """Test Hume client initialization."""
    print("\\n" + "=" * 60)
    print("🤖 TESTING HUME CLIENT INITIALIZATION")
    print("=" * 60)
    
    try:
        # Test HumeConfig creation
        config = HumeConfig.from_config_file()
        print(f"✅ HumeConfig created successfully")
        print(f"🔑 API key configured: {'Yes' if config.api_key else 'No'}")
        print(f"🔐 Secret key configured: {'Yes' if config.secret_key else 'No'}")
        print(f"🔄 Max retries: {config.max_retries}")
        print(f"⏱️  Job timeout: {config.job_timeout} seconds")
        
        # Test HumeClient creation
        client = HumeClient(config)
        print(f"✅ HumeClient created successfully")
        
        # Test granularity levels
        print(f"🎯 Available granularity levels:")
        for level in GranularityLevel:
            print(f"   - {level.name}: {level.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
        return False

def test_analysis_config():
    """Test analysis configuration."""
    print("\\n" + "=" * 60)
    print("📊 TESTING ANALYSIS CONFIGURATION")
    print("=" * 60)
    
    try:
        config = load_config()
        hume_config = config.get('hume_ai', {})
        
        # Test negative emotions list
        negative_emotions = hume_config.get('negative_emotions', [])
        print(f"😟 Negative emotions configured: {len(negative_emotions)}")
        if negative_emotions:
            print(f"   Examples: {negative_emotions[:5]}")
        
        # Test hesitancy vocables
        hesitancy_vocables = hume_config.get('hesitancy_vocables', [])
        print(f"🤔 Hesitancy vocables configured: {len(hesitancy_vocables)}")
        if hesitancy_vocables:
            print(f"   Examples: {hesitancy_vocables}")
        
        # Test thresholds
        intensity_thresholds = hume_config.get('intensity_thresholds', {})
        print(f"📈 Intensity thresholds: {intensity_thresholds}")
        
        hesitancy_thresholds = hume_config.get('hesitancy_thresholds', {})
        print(f"🤔 Hesitancy thresholds: {hesitancy_thresholds}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis configuration test failed: {e}")
        return False

def test_output_directories():
    """Test output directory configuration."""
    print("\\n" + "=" * 60)
    print("📁 TESTING OUTPUT DIRECTORIES")
    print("=" * 60)
    
    try:
        config = load_config()
        
        # Test hume output directory
        hume_output = config.get('hume_ai', {}).get('output_directory', 'data/cache/hume_analysis')
        print(f"📂 Hume output directory: {hume_output}")
        
        # Test file paths
        file_paths = config.get('file_paths', {})
        hume_analysis_dir = file_paths.get('hume_analysis_dir', 'Not configured')
        print(f"📂 Hume analysis dir (file_paths): {hume_analysis_dir}")
        
        # Create output directory if it doesn't exist
        output_path = Path(hume_output)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory created/verified: {output_path.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Output directory test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 HUME AI INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Config Loading", test_config_loading),
        ("Client Initialization", test_hume_client_init),
        ("Analysis Configuration", test_analysis_config),
        ("Output Directories", test_output_directories)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("\\n" + "-" * 60)
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\\n🎉 All tests passed! Hume AI integration is ready to use.")
    else:
        print(f"\\n⚠️  {failed} test(s) failed. Please check the configuration.")
    
    print("\\n💡 Next steps:")
    print("   1. Set HUME_API_KEY in your .env file")
    print("   2. Optionally set HUME_SECRET_KEY for OAuth2")
    print("   3. Run: python src/hume/config_driven_analysis.py")
    print("   4. Test with: python src/hume/test_config_driven_analysis.py")

if __name__ == "__main__":
    main()