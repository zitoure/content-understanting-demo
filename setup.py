#!/usr/bin/env python3
"""
Setup script for Azure AI Content Understanding Demo

This script helps users set up their environment and validate their configuration
before running the demos.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import requests

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is 3.11 or higher."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.11+)"

def install_dependencies() -> Tuple[bool, str]:
    """Install required Python packages."""
    try:
        print("Installing required packages...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True,
            check=True
        )
        return True, "All dependencies installed successfully"
    except subprocess.CalledProcessError as e:
        return False, f"Failed to install dependencies: {e.stderr}"

def create_env_file() -> Tuple[bool, str]:
    """Create .env file from template if it doesn't exist."""
    env_file = Path(".env")
    template_file = Path(".env.template")
    
    if env_file.exists():
        return True, ".env file already exists"
    
    if not template_file.exists():
        return False, ".env.template file not found"
    
    try:
        # Copy template to .env
        with open(template_file, 'r') as template:
            content = template.read()
        
        with open(env_file, 'w') as env:
            env.write(content)
        
        return True, ".env file created from template"
    except Exception as e:
        return False, f"Failed to create .env file: {e}"

def validate_env_config() -> Dict[str, Tuple[bool, str]]:
    """Validate environment configuration."""
    from dotenv import load_dotenv
    load_dotenv()
    
    validations = {}
    
    # Check endpoint
    endpoint = os.getenv("AZURE_AI_SERVICE_ENDPOINT")
    if endpoint:
        if endpoint.startswith("https://") and "cognitiveservices.azure.com" in endpoint:
            validations["endpoint"] = (True, f"Valid endpoint: {endpoint[:50]}...")
        else:
            validations["endpoint"] = (False, "Invalid endpoint format")
    else:
        validations["endpoint"] = (False, "AZURE_AI_SERVICE_ENDPOINT not set")
    
    # Check API key
    api_key = os.getenv("AZURE_AI_SERVICE_API_KEY")
    if api_key:
        if len(api_key) >= 32:  # Azure keys are typically 32+ characters
            validations["api_key"] = (True, f"API key present ({len(api_key)} chars)")
        else:
            validations["api_key"] = (False, "API key appears too short")
    else:
        validations["api_key"] = (False, "AZURE_AI_SERVICE_API_KEY not set (Azure AD auth will be used)")
    
    # Check API version
    api_version = os.getenv("AZURE_AI_SERVICE_API_VERSION")
    if api_version:
        validations["api_version"] = (True, f"API version: {api_version}")
    else:
        validations["api_version"] = (True, "Using default API version")
    
    return validations

def test_azure_connection() -> Tuple[bool, str]:
    """Test connection to Azure AI Service."""
    from dotenv import load_dotenv
    load_dotenv()
    
    endpoint = os.getenv("AZURE_AI_SERVICE_ENDPOINT")
    api_key = os.getenv("AZURE_AI_SERVICE_API_KEY")
    
    if not endpoint:
        return False, "No endpoint configured"
    
    try:
        # Try to list analyzers (this is a simple endpoint that should work)
        headers = {}
        if api_key:
            headers["Ocp-Apim-Subscription-Key"] = api_key
        else:
            return False, "No API key for connection test"
        
        url = f"{endpoint.rstrip('/')}/contentunderstanding/analyzers"
        params = {"api-version": "2025-05-01-preview"}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            return True, "Successfully connected to Azure AI Service"
        elif response.status_code == 401:
            return False, "Authentication failed - check your API key"
        elif response.status_code == 403:
            return False, "Access forbidden - check your subscription and permissions"
        else:
            return False, f"Connection failed with status {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, "Connection timed out"
    except requests.exceptions.ConnectionError:
        return False, "Connection error - check your endpoint URL"
    except Exception as e:
        return False, f"Connection test failed: {e}"

def create_sample_data_dir():
    """Create sample data directory with information."""
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    readme_content = """# Sample Data Directory

This directory is for your sample files to test with Content Understanding.

## Supported File Types

### Documents
- PDF files (.pdf)
- Images (.jpg, .png, .bmp, .heif, .tiff)
- Office documents (.docx, .xlsx, .pptx)
- Text files (.txt, .html, .md, .rtf, .xml)

### Audio
- WAV files (.wav)
- MP3 files (.mp3)
- Other formats (.mp4, .opus, .ogg, .flac, .wma, .aac, .amr, .3gp, .webm, .m4a, .spx)

### Size Limits
- Documents: ≤ 200 MB, ≤ 300 pages
- Audio: ≤ 1 GB, ≤ 4 hours
- Images: ≤ 200 MB, 50x50 to 10k x 10k pixels

## Usage

1. Place your sample files in this directory
2. Update the demo scripts to use your local files instead of URLs
3. Run the demos with your own data

## Example File URLs (Used in demos)
- Invoice PDF: https://github.com/Azure-Samples/azure-ai-content-understanding-python/raw/refs/heads/main/data/invoice.pdf
- Receipt Image: https://github.com/Azure-Samples/azure-ai-content-understanding-python/raw/refs/heads/main/data/receipt.png
- Audio File: https://github.com/Azure-Samples/azure-ai-content-understanding-python/raw/refs/heads/main/data/audio.wav
"""
    
    readme_path = sample_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

def main():
    """Main setup function."""
    print("🔧 Azure AI Content Understanding Demo Setup")
    print("=" * 50)
    
    # Check Python version
    print("1. Checking Python version...")
    python_ok, python_msg = check_python_version()
    status = "✅" if python_ok else "❌"
    print(f"   {status} {python_msg}")
    
    if not python_ok:
        print("\n❌ Setup failed: Python 3.11+ is required")
        return False
    
    # Install dependencies
    print("\n2. Installing dependencies...")
    deps_ok, deps_msg = install_dependencies()
    status = "✅" if deps_ok else "❌"
    print(f"   {status} {deps_msg}")
    
    if not deps_ok:
        print("\n❌ Setup failed: Could not install dependencies")
        return False
    
    # Create .env file
    print("\n3. Setting up environment configuration...")
    env_ok, env_msg = create_env_file()
    status = "✅" if env_ok else "❌"
    print(f"   {status} {env_msg}")
    
    # Validate configuration
    print("\n4. Validating configuration...")
    validations = validate_env_config()
    
    all_valid = True
    for check, (valid, msg) in validations.items():
        status = "✅" if valid else "❌"
        print(f"   {status} {check}: {msg}")
        if not valid and check in ["endpoint"]:  # Critical checks
            all_valid = False
    
    # Test connection
    if all_valid:
        print("\n5. Testing Azure connection...")
        conn_ok, conn_msg = test_azure_connection()
        status = "✅" if conn_ok else "❌"
        print(f"   {status} {conn_msg}")
        
        if not conn_ok:
            all_valid = False
    
    # Create sample data directory
    print("\n6. Setting up sample data directory...")
    create_sample_data_dir()
    print("   ✅ Sample data directory created")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    print("   ✅ Output directory created")
    
    # Final status
    print("\n" + "=" * 50)
    if all_valid:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Update your .env file with your Azure credentials")
        print("2. Run the comprehensive demo: python comprehensive_demo.py")
        print("3. Try individual demos in document_analysis_demo.py or audio_analysis_demo.py")
        print("\n📁 Directories created:")
        print("   - ./output/ (for demo results)")
        print("   - ./sample_data/ (for your test files)")
    else:
        print("⚠️  Setup completed with warnings!")
        print("\nPlease fix the configuration issues above before running demos:")
        print("1. Edit the .env file with your Azure AI Service details")
        print("2. Ensure your Azure subscription is active")
        print("3. Verify your API key and endpoint are correct")
    
    return all_valid

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
