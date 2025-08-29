#!/usr/bin/env python3
"""
Test Config-Driven Analysis

Test script to verify the config-driven analysis setup and configuration.
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

from hume.config_driven_analysis import ConfigDrivenHumeAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)

def test_config_loading():
    """Test configuration loading and validation."""
    print("=" * 60)
    print("🔧 TESTING CONFIG-DRIVEN ANALYSIS SETUP")
    print("=" * 60)
    
    try:
        analyzer = ConfigDrivenHumeAnalyzer()
        
        print(f"✅ Configuration loaded successfully")
        print(f"📊 Score threshold: {analyzer.score_threshold}")
        print(f"🔢 Top N emotions: {analyzer.top_n_emotions}")
        print(f"😟 Negative emotions count: {len(analyzer.negative_emotions)}")
        print(f"🤔 Hesitancy vocables count: {len(analyzer.hesitancy_vocables)}")
        print(f"📁 Processed audio dir: {analyzer.processed_audio_dir}")
        print(f"📁 Hume analysis base dir: {analyzer.hume_analysis_dir}")
        print(f"📁 Campaign structure: <base_dir>/<campaign>_reps/<FirstName_LastName>/<contact_id>/")
        print(f"📄 Result filename: <contact_id>_analysis_results.json")
        print(f"🎵 Audio concatenation: Segments combined into single conversation file")
        
        auto_discover = analyzer.config.get('hume_ai', {}).get('auto_discover_segments', False)
        print(f"🔍 Auto-discover segments: {'Enabled' if auto_discover else 'Disabled'}")
        print(f"💡 Use 'segments': [\"ALL\"] to auto-discover all segments in a contact directory")
        print(f"🎯 Default granularity: {analyzer.default_granularity.value}")
        
        # Test analysis selection
        analysis_selection = analyzer.hume_config.get('analysis_selection', {})
        print(f"\\n📋 Analysis Selection:")
        for campaign, campaign_data in analysis_selection.items():
            print(f"  🏢 Campaign: {campaign}")
            for sales_rep, rep_data in campaign_data.items():
                print(f"    👤 Sales Rep: {sales_rep}")
                for contact, contact_data in rep_data.items():
                    segments = contact_data.get('segments', [])
                    print(f"      📞 Contact: {contact}")
                    print(f"        🎵 Segments: {segments}")
        
        # Test combination method
        combination_method = analyzer.hume_config.get('combination_method', 'ordered')
        print(f"\\n🔄 Combination method: {combination_method}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_segment_path_resolution():
    """Test segment path resolution."""
    print("\\n" + "=" * 60)
    print("📁 TESTING SEGMENT PATH RESOLUTION")
    print("=" * 60)
    
    try:
        analyzer = ConfigDrivenHumeAnalyzer()
        analysis_selection = analyzer.hume_config.get('analysis_selection', {})
        
        found_segments = 0
        missing_segments = 0
        
        for campaign, campaign_data in analysis_selection.items():
            for sales_rep, rep_data in campaign_data.items():
                for contact, contact_data in rep_data.items():
                    segments = contact_data.get('segments', [])
                    
                    print(f"\\n🔍 Checking: {campaign}/{sales_rep}/{contact}")
                    segment_paths = analyzer.get_segment_paths(campaign, sales_rep, contact, segments)
                    
                    for segment in segments:
                        # Fix sales_rep format: replace \ with _ for proper path format
                        formatted_sales_rep = sales_rep.replace('\\', '_')
                        base_path = analyzer.processed_audio_dir / campaign / formatted_sales_rep / contact
                        segment_path = base_path / segment
                        
                        if segment_path.exists():
                            print(f"  ✅ Found: {segment_path}")
                            found_segments += 1
                        else:
                            print(f"  ❌ Missing: {segment_path}")
                            missing_segments += 1
        
        print(f"\\n📊 Path Resolution Summary:")
        print(f"  ✅ Found segments: {found_segments}")
        print(f"  ❌ Missing segments: {missing_segments}")
        
        if missing_segments > 0:
            print(f"\\n⚠️  Warning: {missing_segments} segments are missing")
            print("   Make sure audio processing has been completed")
        
        return found_segments > 0
        
    except Exception as e:
        print(f"❌ Path resolution test failed: {e}")
        return False

def test_hume_client_setup():
    """Test Hume client configuration."""
    print("\\n" + "=" * 60)
    print("🤖 TESTING HUME CLIENT SETUP")
    print("=" * 60)
    
    try:
        analyzer = ConfigDrivenHumeAnalyzer()
        
        print(f"✅ Hume client initialized")
        print(f"🔑 API key configured: {'Yes' if analyzer.hume_client_config.api_key else 'No'}")
        print(f"🔐 Secret key configured: {'Yes' if analyzer.hume_client_config.secret_key else 'No'}")
        print(f"🔄 Max retries: {analyzer.hume_client_config.max_retries}")
        print(f"⏱️  Job timeout: {analyzer.hume_client_config.job_timeout} seconds")
        
        if not analyzer.hume_client_config.api_key:
            print("\\n⚠️  Warning: HUME_API_KEY not set in .env file")
            print("   Set your API key before running analysis")
        
        return True
        
    except Exception as e:
        print(f"❌ Hume client setup test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 CONFIG-DRIVEN HUME ANALYSIS TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Segment Path Resolution", test_segment_path_resolution),
        ("Hume Client Setup", test_hume_client_setup)
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
        print("\\n🎉 All tests passed! Config-driven analysis is ready to use.")
        print("\\n🚀 To run analysis:")
        print("   python src/hume/config_driven_analysis.py")
    else:
        print(f"\\n⚠️  {failed} test(s) failed. Please check the configuration.")
    
    print("\\n💡 Configuration Tips:")
    print("   1. Ensure HUME_API_KEY is set in .env file")
    print("   2. Run audio processing first to generate segments")
    print("   3. Update analysis_selection in audio_analysis.json as needed")
    print("   4. Choose combination_method: 'ordered', 'random', or 'specific'")

if __name__ == "__main__":
    main()