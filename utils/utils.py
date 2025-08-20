"""
Utility functions for Azure AI Content Understanding demos.
Provides common functionality for file handling, validation, and result processing.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
from urllib.parse import urlparse
import requests


def setup_logging(name: str = "demo", level: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a logger for demo scripts.

    Args:
        name: Logger name
        level: Optional log level string (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logging.Logger instance
    """
    # Determine level from argument or environment
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    logger = logging.getLogger(name)
    # Prevent adding duplicate handlers if logger already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(numeric_level)

    return logger


def create_output_directory(subdir: str = "output") -> Path:
    """
    Create (if needed) and return a Path to an output directory under the
    DEFAULT_OUTPUT_DIR environment variable or the repository root.

    Args:
        subdir: Subdirectory name under the default output directory.

    Returns:
        pathlib.Path for the created directory
    """
    base_dir = os.getenv("DEFAULT_OUTPUT_DIR", "./output")
    out_dir = Path(base_dir) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def validate_file_size(file_path: str, max_size_mb: int = 200) -> bool:
    """
    Validate that a file is within size limits.
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum file size in MB
        
    Returns:
        True if file is within limits
    """
    try:
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        return file_size_mb <= max_size_mb
    except OSError:
        return False


def get_file_type(file_path: str) -> str:
    """
    Get the file type category based on extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File type category (document, image, audio, video, unknown)
    """
    ext = Path(file_path).suffix.lower()
    
    document_exts = {'.pdf', '.docx', '.xlsx', '.pptx', '.txt', '.html', '.md', '.rtf', '.xml'}
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.heif', '.tiff', '.tif'}
    audio_exts = {'.wav', '.mp3', '.mp4', '.opus', '.ogg', '.flac', '.wma', '.aac', '.amr', '.3gp', '.webm', '.m4a', '.spx'}
    video_exts = {'.mp4', '.m4v', '.flv', '.wmv', '.asf', '.avi', '.mkv', '.mov'}
    
    if ext in document_exts:
        return 'document'
    elif ext in image_exts:
        return 'image'
    elif ext in audio_exts:
        return 'audio'
    elif ext in video_exts:
        return 'video'
    else:
        return 'unknown'


def validate_url(url: str) -> bool:
    """
    Validate that a URL is properly formed and accessible.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid and accessible
    """
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False
        
        # Try to access the URL
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except:
        return False


def create_sample_analyzer_configs() -> Dict[str, Dict[str, Any]]:
    """
    Create sample analyzer configurations for different use cases.
    
    Returns:
        Dictionary of analyzer configurations
    """
    configs = {
        "invoice_analyzer": {
            "description": "Invoice processing analyzer",
            "baseAnalyzerId": "prebuilt-documentAnalyzer",
            "config": {
                "returnDetails": True,
                "enableFormula": False,
                "disableContentFiltering": False,
                "estimateFieldSourceAndConfidence": True,
                "tableFormat": "html"
            },
            "fieldSchema": {
                "fields": {
                    "VendorName": {
                        "type": "string",
                        "description": "Name of the vendor or supplier"
                    },
                    "InvoiceNumber": {
                        "type": "string",
                        "description": "Invoice number or ID"
                    },
                    "InvoiceDate": {
                        "type": "date",
                        "description": "Date of the invoice"
                    },
                    "DueDate": {
                        "type": "date",
                        "description": "Payment due date"
                    },
                    "TotalAmount": {
                        "type": "number",
                        "description": "Total amount to be paid"
                    },
                    "Currency": {
                        "type": "string",
                        "description": "Currency of the invoice"
                    },
                    "LineItems": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Description": {"type": "string"},
                                "Quantity": {"type": "number"},
                                "UnitPrice": {"type": "number"},
                                "Amount": {"type": "number"}
                            }
                        },
                        "description": "List of line items in the invoice"
                    }
                }
            }
        },
        
        "contract_analyzer": {
            "description": "Contract analysis analyzer",
            "baseAnalyzerId": "prebuilt-documentAnalyzer",
            "config": {
                "returnDetails": True,
                "enableFormula": False,
                "disableContentFiltering": False,
                "estimateFieldSourceAndConfidence": True,
                "tableFormat": "html"
            },
            "fieldSchema": {
                "fields": {
                    "ContractType": {
                        "type": "string",
                        "enum": ["Service Agreement", "Purchase Agreement", "Employment Contract", "NDA", "Other"],
                        "description": "Type of contract"
                    },
                    "Parties": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Parties involved in the contract"
                    },
                    "EffectiveDate": {
                        "type": "date",
                        "description": "Contract effective date"
                    },
                    "ExpirationDate": {
                        "type": "date",
                        "description": "Contract expiration date"
                    },
                    "ContractValue": {
                        "type": "number",
                        "description": "Total contract value"
                    },
                    "KeyTerms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key terms and conditions"
                    },
                    "RenewalClause": {
                        "type": "boolean",
                        "description": "Whether contract has renewal clause"
                    }
                }
            }
        },
        
        "meeting_analyzer": {
            "description": "Meeting audio analyzer",
            "baseAnalyzerId": "prebuilt-audioAnalyzer",
            "config": {
                "locales": ["en-US"],
                "returnDetails": True,
                "disableContentFiltering": False
            },
            "fieldSchema": {
                "fields": {
                    "MeetingType": {
                        "type": "string",
                        "enum": ["Team Meeting", "Client Call", "Interview", "Training", "Other"],
                        "description": "Type of meeting"
                    },
                    "Attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Meeting attendees identified"
                    },
                    "KeyDecisions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key decisions made during the meeting"
                    },
                    "ActionItems": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Task": {"type": "string"},
                                "Owner": {"type": "string"},
                                "DueDate": {"type": "string"}
                            }
                        },
                        "description": "Action items assigned during meeting"
                    },
                    "NextMeeting": {
                        "type": "string",
                        "description": "Date/time of next meeting if mentioned"
                    },
                    "Sentiment": {
                        "type": "string",
                        "enum": ["Positive", "Neutral", "Negative"],
                        "description": "Overall meeting sentiment"
                    }
                }
            }
        },
        
        "customer_service_analyzer": {
            "description": "Customer service call analyzer",
            "baseAnalyzerId": "prebuilt-callCenter",
            "config": {
                "locales": ["en-US", "es-ES"],
                "returnDetails": True,
                "disableContentFiltering": False
            },
            "fieldSchema": {
                "fields": {
                    "CallPurpose": {
                        "type": "string",
                        "enum": ["Complaint", "Inquiry", "Support", "Sales", "Billing", "Other"],
                        "description": "Primary purpose of the call"
                    },
                    "CustomerSatisfaction": {
                        "type": "string",
                        "enum": ["Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied"],
                        "description": "Customer satisfaction level"
                    },
                    "IssueResolved": {
                        "type": "boolean",
                        "description": "Whether the issue was resolved"
                    },
                    "EscalationRequired": {
                        "type": "boolean",
                        "description": "Whether escalation is required"
                    },
                    "ProductsDiscussed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Products or services discussed"
                    },
                    "FollowUpDate": {
                        "type": "string",
                        "description": "Follow-up date if scheduled"
                    },
                    "AgentNotes": {
                        "type": "string",
                        "method": "generate",
                        "description": "Summary notes for agent reference"
                    }
                }
            }
        }
    }
    
    return configs


def format_analysis_results(result: Dict[str, Any], output_format: str = "summary") -> str:
    """
    Format analysis results for display.
    
    Args:
        result: Raw analysis results
        output_format: Format type (summary, detailed, json)
        
    Returns:
        Formatted results string
    """
    if output_format == "json":
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    status = result.get("status", "Unknown")
    analyzer_id = result.get("result", {}).get("analyzerId", "Unknown")
    created_at = result.get("result", {}).get("createdAt", "Unknown")
    
    output = f"Analysis Results\n"
    output += f"================\n"
    output += f"Status: {status}\n"
    output += f"Analyzer: {analyzer_id}\n"
    output += f"Created: {created_at}\n\n"
    
    contents = result.get("result", {}).get("contents", [])
    if contents:
        content = contents[0]
        
        # Extract basic content info
        if "markdown" in content:
            markdown_preview = content["markdown"][:300] + "..." if len(content["markdown"]) > 300 else content["markdown"]
            output += f"Content Preview:\n{markdown_preview}\n\n"
        
        # Extract fields
        fields = content.get("fields", {})
        if fields:
            output += "Extracted Fields:\n"
            output += "-" * 16 + "\n"
            for field_name, field_data in fields.items():
                field_type = field_data.get("type", "unknown")
                
                if field_type == "string":
                    value = field_data.get("valueString", "")
                elif field_type == "number":
                    value = field_data.get("valueNumber", "")
                elif field_type == "date":
                    value = field_data.get("valueDate", "")
                elif field_type == "boolean":
                    value = field_data.get("valueBoolean", "")
                elif field_type == "array":
                    value = field_data.get("valueArray", [])
                    if isinstance(value, list) and len(value) > 3:
                        value = f"{', '.join(map(str, value[:3]))}... ({len(value)} total)"
                    else:
                        value = ', '.join(map(str, value))
                else:
                    value = str(field_data.get("value", ""))
                
                output += f"{field_name}: {value}\n"
        
        # Extract structure info for documents
        if output_format == "detailed":
            structure_info = []
            if "paragraphs" in content:
                structure_info.append(f"Paragraphs: {len(content['paragraphs'])}")
            if "tables" in content:
                structure_info.append(f"Tables: {len(content['tables'])}")
            if "sections" in content:
                structure_info.append(f"Sections: {len(content['sections'])}")
            
            if structure_info:
                output += f"\nDocument Structure:\n"
                output += "-" * 18 + "\n"
                output += "\n".join(structure_info) + "\n"
        
        # Extract warnings
        warnings = result.get("result", {}).get("warnings", [])
        if warnings:
            output += f"\nWarnings:\n"
            output += "-" * 9 + "\n"
            for warning in warnings:
                output += f"- {warning}\n"
    
    return output


def save_results_to_file(
    results: Dict[str, Any], 
    output_path: Union[str, Path],
    format_type: str = "json"
) -> bool:
    """
    Save analysis results to file.
    
    Args:
        results: Analysis results dictionary
        output_path: Output file path
        format_type: Output format (json, summary, detailed)
        
    Returns:
        True if successful
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            formatted_results = format_analysis_results(results, format_type)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_results)
        
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False


def create_batch_processing_config(
    file_urls: List[str],
    analyzer_id: str,
    output_dir: str = "./batch_output"
) -> Dict[str, Any]:
    """
    Create configuration for batch processing multiple files.
    
    Args:
        file_urls: List of file URLs to process
        analyzer_id: Analyzer to use for processing
        output_dir: Directory for output files
        
    Returns:
        Batch processing configuration
    """
    config = {
        "analyzer_id": analyzer_id,
        "files": [],
        "output_dir": output_dir,
        "concurrent_requests": 5,
        "timeout_per_file": 300
    }
    
    for i, url in enumerate(file_urls):
        file_config = {
            "url": url,
            "output_file": f"result_{i+1:03d}.json",
            "file_type": get_file_type(url)
        }
        config["files"].append(file_config)
    
    return config


def extract_transcript_text(result: Dict[str, Any]) -> str:
    """
    Extract clean transcript text from audio/video analysis results.
    
    Args:
        result: Analysis results containing transcript data
        
    Returns:
        Clean transcript text
    """
    contents = result.get("result", {}).get("contents", [])
    if not contents:
        return ""
    
    content = contents[0]
    
    # Try to extract from transcript phrases first
    transcript_phrases = content.get("transcriptPhrases", [])
    if transcript_phrases:
        transcript_lines = []
        for phrase in transcript_phrases:
            text = phrase.get("text", "").strip()
            if text:
                transcript_lines.append(text)
        return " ".join(transcript_lines)
    
    # Fallback to markdown extraction
    markdown = content.get("markdown", "")
    if "WEBVTT" in markdown:
        lines = markdown.split('\n')
        transcript_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip WEBVTT headers, timestamps, and empty lines
            if (line and 
                not line.startswith('WEBVTT') and 
                '-->' not in line and 
                not line.startswith('NOTE') and
                not line.startswith('#')):
                
                # Clean speaker tags
                if line.startswith('<v ') and '>' in line:
                    line = line[line.index('>') + 1:].strip()
                
                if line:
                    transcript_lines.append(line)
        
        return " ".join(transcript_lines)
    
    return ""


def validate_azure_config() -> Dict[str, bool]:
    """
    Validate Azure configuration from environment variables.
    
    Returns:
        Dictionary with validation results
    """
    validation = {
        "endpoint": bool(os.getenv("AZURE_AI_SERVICE_ENDPOINT")),
        "api_key": bool(os.getenv("AZURE_AI_SERVICE_API_KEY")),
        "api_version": bool(os.getenv("AZURE_AI_SERVICE_API_VERSION")),
        "output_dir": True,  # Always valid as we create it
    }
    
    # Check if endpoint is properly formatted
    endpoint = os.getenv("AZURE_AI_SERVICE_ENDPOINT", "")
    if endpoint:
        validation["endpoint_format"] = endpoint.startswith("https://") and endpoint.endswith(".services.ai.azure.com/")
    else:
        validation["endpoint_format"] = False
    
    return validation
