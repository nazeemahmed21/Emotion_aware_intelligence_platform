#!/usr/bin/env python3
"""
Hume AI Setup Helper
Helps you configure Hume AI credentials and test the integration
"""
import os
import sys
from pathlib import Path

def create_env_file():
    """Create or update .env file with Hume AI credentials"""
    print("🔧 Hume AI Setup Helper")
    print("=" * 30)
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    # Read existing .env if it exists
    existing_vars = {}
    if env_file.exists():
        print("📁 Found existing .env file")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    existing_vars[key] = value
    else:
        print("📁 No .env file found, creating new one")
    
    # Get Hume AI credentials
    print("\n🎭 Hume AI Credentials")
    print("Get these from: https://platform.hume.ai/")
    print("-" * 40)
    
    # API Key (required)
    current_api_key = existing_vars.get('HUME_API_KEY', '')
    if current_api_key:
        print(f"Current API Key: {current_api_key[:8]}...{current_api_key[-4:] if len(current_api_key) > 12 else current_api_key}")
        use_existing = input("Keep existing API key? (y/n): ").lower().strip()
        if use_existing == 'y':
            api_key = current_api_key
        else:
            api_key = input("Enter your Hume AI API Key: ").strip()
    else:
        api_key = input("Enter your Hume AI API Key: ").strip()
    
    if not api_key:
        print("❌ API Key is required!")
        return False
    
    # Secret Key (optional)
    current_secret_key = existing_vars.get('HUME_SECRET_KEY', '')
    if current_secret_key:
        print(f"Current Secret Key: {current_secret_key[:8]}...{current_secret_key[-4:] if len(current_secret_key) > 12 else current_secret_key}")
        use_existing = input("Keep existing Secret key? (y/n): ").lower().strip()
        if use_existing == 'y':
            secret_key = current_secret_key
        else:
            secret_key = input("Enter your Hume AI Secret Key (optional, press Enter to skip): ").strip()
    else:
        secret_key = input("Enter your Hume AI Secret Key (optional, press Enter to skip): ").strip()
    
    # Webhook URL (optional)
    current_webhook = existing_vars.get('HUME_WEBHOOK_URL', '')
    if current_webhook:
        print(f"Current Webhook URL: {current_webhook}")
        use_existing = input("Keep existing webhook URL? (y/n): ").lower().strip()
        if use_existing == 'y':
            webhook_url = current_webhook
        else:
            webhook_url = input("Enter webhook URL (optional, press Enter to skip): ").strip()
    else:
        webhook_url = input("Enter webhook URL (optional, press Enter to skip): ").strip()
    
    # Write .env file
    print("\n💾 Writing .env file...")
    
    # Start with existing content or example
    env_content = []
    
    if env_example.exists():
        with open(env_example, 'r') as f:
            env_content = f.readlines()
    
    # Update or add Hume AI variables
    hume_vars = {
        'HUME_API_KEY': api_key,
        'HUME_SECRET_KEY': secret_key,
        'HUME_WEBHOOK_URL': webhook_url
    }
    
    # Update existing content
    updated_content = []
    hume_vars_added = set()
    
    for line in env_content:
        line = line.rstrip()
        if '=' in line and not line.startswith('#'):
            key = line.split('=')[0]
            if key in hume_vars:
                if hume_vars[key]:  # Only add if not empty
                    updated_content.append(f"{key}={hume_vars[key]}")
                    hume_vars_added.add(key)
                # Skip empty values
            else:
                updated_content.append(line)
        else:
            updated_content.append(line)
    
    # Add any missing Hume variables
    for key, value in hume_vars.items():
        if key not in hume_vars_added and value:
            updated_content.append(f"{key}={value}")
    
    # Write the file
    with open(env_file, 'w') as f:
        for line in updated_content:
            f.write(line + '\n')
    
    print(f"✅ .env file updated with Hume AI credentials")
    return True

def test_setup():
    """Test the Hume AI setup"""
    print("\n🧪 Testing Hume AI Setup")
    print("=" * 30)
    
    # Check if test file exists
    test_file = Path("test_hume_integration.py")
    if not test_file.exists():
        print("❌ test_hume_integration.py not found")
        return False
    
    # Run the test
    print("Running integration tests...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "test_hume_integration.py"], 
                              capture_output=True, text=True)
        
        print("Test output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    """Main setup function"""
    print("🎭 Welcome to Hume AI Integration Setup!")
    print("=" * 50)
    
    # Step 1: Create .env file
    if not create_env_file():
        print("\n❌ Setup failed at credential configuration")
        return False
    
    # Step 2: Test the setup
    print("\n" + "=" * 50)
    test_choice = input("Would you like to test the integration now? (y/n): ").lower().strip()
    
    if test_choice == 'y':
        if test_setup():
            print("\n🎉 Setup completed successfully!")
            print("\n🚀 Next steps:")
            print("1. Run the main app: python run.py")
            print("2. Upload audio files to test Hume AI emotion recognition")
            print("3. Check the HUME_INTEGRATION.md file for more details")
        else:
            print("\n⚠️  Setup completed but tests failed")
            print("Check your API credentials and internet connection")
    else:
        print("\n✅ Setup completed!")
        print("Run 'python test_hume_integration.py' when ready to test")
    
    print("\n📚 For more information, see HUME_INTEGRATION.md")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)