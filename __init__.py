"""
Azure AI Content Understanding Demo Package

A comprehensive demonstration of Azure AI Content Understanding capabilities
including document analysis, audio processing, and custom analyzer creation.

Available demos:
- comprehensive_demo.py: Full feature demonstration
- document_analysis_demo.py: Document processing examples  
- audio_analysis_demo.py: Audio analysis examples
- batch_processing_demo.py: Batch processing capabilities
- rag_integration_demo.py: RAG system integration
- setup.py: Environment setup and validation

Quick start:
1. Run: python setup.py
2. Configure your .env file with Azure credentials
3. Run: python comprehensive_demo.py
"""

from content_understanding_client import AzureContentUnderstandingClient, create_client_from_env, PREBUILT_ANALYZERS
from document_analysis_demo import DocumentAnalyzer
from audio_analysis_demo import AudioAnalyzer
from utils import (
    validate_file_size,
    get_file_type,
    validate_url,
    format_analysis_results,
    create_sample_analyzer_configs,
    validate_azure_config
)

__version__ = "1.0.0"
__author__ = "Azure AI Content Understanding Team"

# Export main classes and functions
__all__ = [
    'AzureContentUnderstandingClient',
    'create_client_from_env',
    'DocumentAnalyzer',
    'AudioAnalyzer',
    'PREBUILT_ANALYZERS',
    'validate_azure_config',
    'format_analysis_results'
]

def get_demo_info():
    """Get information about available demos."""
    return {
        "demos": {
            "comprehensive_demo.py": {
                "description": "Complete demonstration of all features",
                "features": ["Document analysis", "Audio processing", "Custom analyzers", "Management operations"],
                "estimated_time": "5-10 minutes"
            },
            "document_analysis_demo.py": {
                "description": "Document processing with prebuilt and custom analyzers",
                "features": ["PDF analysis", "Image processing", "Custom field extraction"],
                "estimated_time": "3-5 minutes"
            },
            "audio_analysis_demo.py": {
                "description": "Audio transcription and analysis",
                "features": ["Speech-to-text", "Speaker diarization", "Call center analysis"],
                "estimated_time": "5-8 minutes"
            },
            "batch_processing_demo.py": {
                "description": "Process multiple files concurrently",
                "features": ["Concurrent processing", "Error handling", "Progress tracking"],
                "estimated_time": "3-5 minutes"
            },
            "rag_integration_demo.py": {
                "description": "Integration with RAG systems",
                "features": ["Content chunking", "Vector preparation", "Knowledge base creation"],
                "estimated_time": "5-7 minutes"
            }
        },
        "setup": {
            "script": "setup.py",
            "description": "Environment setup and configuration validation",
            "estimated_time": "2-3 minutes"
        }
    }

if __name__ == "__main__":
    import json
    demo_info = get_demo_info()
    print("Azure AI Content Understanding Demo Package")
    print("=" * 50)
    print(json.dumps(demo_info, indent=2))
