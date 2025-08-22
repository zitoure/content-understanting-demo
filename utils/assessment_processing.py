"""
Shared assessment processing utilities for Azure AI Content Understanding

This module provides common functionality for processing assessment results
from both healthcare and golf coaching scenarios.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


def process_assessment_results(
    raw_result: Dict[str, Any],
    assessment_type: str,
    template: Dict[str, Any],
    primary_id: Optional[str] = None,
    secondary_id: Optional[str] = None,
    session_metadata: Optional[Dict[str, Any]] = None,
    id_labels: Tuple[str, str] = ("primary_id", "secondary_id"),
    use_detailed_structure: bool = False
) -> Dict[str, Any]:
    """
    Process raw Azure AI Content Understanding results into structured assessment format.
    
    This function works for both healthcare and golf coaching assessments with flexible
    naming and structure options.
    
    Args:
        raw_result: Raw analysis result from Azure AI Content Understanding
        assessment_type: Type of assessment being processed
        template: Assessment template with field definitions
        primary_id: Primary identifier (patient_id for healthcare, player_id for golf)
        secondary_id: Secondary identifier (care_staff_id for healthcare, coach_id for golf)
        session_metadata: Optional session metadata
        id_labels: Tuple of (primary_label, secondary_label) for the IDs
        use_detailed_structure: If True, use healthcare-style detailed structure
        
    Returns:
        Structured assessment result dictionary
    """
    
    # Extract conversation transcript
    from utils.utils import extract_transcript_text
    transcript = extract_transcript_text(raw_result)
    
    # Extract analysis results
    contents = raw_result.get("result", {}).get("contents", [])
    extracted_fields = {}
    
    if contents:
        content = contents[0]
        fields = content.get("fields", {})
        
        for field_name, field_data in fields.items():
            extracted_value = _extract_field_value(field_data)
            
            # Store the extracted value
            extracted_fields[field_name] = extracted_value
    
    # Create base assessment result structure
    primary_label, secondary_label = id_labels
    
    if use_detailed_structure:
        # Healthcare-style detailed structure
        assessment_result = {
            "assessment_metadata": {
                "assessment_type": assessment_type,
                "assessment_name": template["name"],
                "template_version": template["version"],
                "analysis_timestamp": datetime.now().isoformat(),
                primary_label: primary_id,
                secondary_label: secondary_id,
                "session_metadata": session_metadata or {}
            },
            "conversation_data": {
                "transcript": transcript,
                "duration_info": _extract_duration_info(contents[0] if contents else {}),
                "speaker_info": _extract_speaker_info(contents[0] if contents else {})
            },
            "assessment_responses": {},
            "raw_analysis_result": raw_result
        }
        
        # Map extracted fields to assessment questions (healthcare style)
        for field_name, field_config in template["fields"].items():
            question_type = field_config.get("question_type", "open_ended")
            question_text = field_config.get("question", "")
            extracted_value = extracted_fields.get(field_name)
            
            assessment_response = {
                "question": question_text,
                "question_type": question_type,
                "extracted_value": extracted_value,
                "requires_review": extracted_value is None or extracted_value == "",
                "field_description": field_config.get("description", "")
            }
            
            # Add question-type specific information
            if question_type == "single_choice" and "enum" in field_config:
                assessment_response["possible_choices"] = field_config["enum"]
            elif question_type == "multiple_choice" and "possible_values" in field_config:
                assessment_response["possible_choices"] = field_config["possible_values"]
            elif question_type == "numeric_scale" and "range" in field_config:
                assessment_response["scale_range"] = field_config["range"]
            
            assessment_result["assessment_responses"][field_name] = assessment_response
    
    else:
        # Golf-style simpler structure
        assessment_result = {
            "assessment_info": {
                "assessment_type": assessment_type,
                "template_name": template["name"],
                "template_version": template["version"],
                "analysis_timestamp": datetime.now().isoformat(),
                primary_label: primary_id,
                secondary_label: secondary_id,
                "session_metadata": session_metadata or {}
            },
            "conversation": {
                "transcript": transcript,
                "audio_url": raw_result.get("audio_url", ""),
                "duration_seconds": raw_result.get("duration_seconds")
            },
            "extracted_fields": extracted_fields,
            "raw_analysis": raw_result
        }
    
    return assessment_result


def _extract_field_value(field_data: Dict[str, Any]) -> Any:
    """Extract value from a field data structure."""
    field_type = field_data.get("type")
    extracted_value = None
    
    if field_type == "string":
        extracted_value = field_data.get("valueString")
    elif field_type == "array":
        # Handle array of values (each item might be a complex object)
        array_values = field_data.get("valueArray", [])
        extracted_values = []
        for item in array_values:
            if isinstance(item, dict) and item.get("type") == "string":
                extracted_values.append(item.get("valueString", ""))
            elif isinstance(item, str):
                extracted_values.append(item)
            else:
                extracted_values.append(str(item))
        extracted_value = extracted_values
    elif field_type == "boolean":
        extracted_value = field_data.get("valueBoolean")
    elif field_type == "integer":
        extracted_value = field_data.get("valueInteger")
    elif field_type == "number":
        extracted_value = field_data.get("valueNumber")
    
    return extracted_value


def _extract_duration_info(content: Dict[str, Any]) -> Dict[str, Any]:
    """Extract conversation duration information."""
    duration_info = {}
    
    if "startTimeMs" in content and "endTimeMs" in content:
        start_ms = content["startTimeMs"]
        end_ms = content["endTimeMs"]
        duration_ms = end_ms - start_ms
        
        duration_info = {
            "start_time_ms": start_ms,
            "end_time_ms": end_ms,
            "duration_ms": duration_ms,
            "duration_minutes": duration_ms / 60000
        }
    
    return duration_info


def _extract_speaker_info(content: Dict[str, Any]) -> Dict[str, Any]:
    """Extract speaker information from conversation."""
    speaker_info = {
        "speakers_detected": [],
        "speaker_segments": []
    }
    
    transcript_phrases = content.get("transcriptPhrases", [])
    speakers = set()
    
    for phrase in transcript_phrases:
        speaker = phrase.get("speaker", "Unknown")
        speakers.add(speaker)
        
        speaker_info["speaker_segments"].append({
            "speaker": speaker,
            "text": phrase.get("text", ""),
            "start_time_ms": phrase.get("startTimeMs", 0),
            "duration_ms": phrase.get("endTimeMs", 0) - phrase.get("startTimeMs", 0)
        })
    
    speaker_info["speakers_detected"] = list(speakers)
    speaker_info["speaker_count"] = len(speakers)
    
    return speaker_info
