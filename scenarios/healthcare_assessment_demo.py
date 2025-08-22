"""
Healthcare Patient Assessment Demo for Azure AI Content Understanding

This system demonstrates how care staff conversations with patients can be automatically
analyzed to fill out standardized assessment forms using AI content understanding.

Features:
- Audio transcription and analysis of patient conversations
- Automatic extraction of assessment data
- Support for multiple question types (open-ended, single choice, multiple choice)
- Extraction scoring for extracted information
- HIPAA-compliant data handling practices
"""

import os
import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from content_understanding_client import AzureContentUnderstandingClient, create_client_from_env
from utils.utils import extract_transcript_text

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PatientAssessmentAnalyzer:
    """
    Analyzes patient-care staff conversations to automatically fill assessment forms.
    Designed for healthcare settings with appropriate privacy and accuracy considerations.
    """
    
    def __init__(self, client: AzureContentUnderstandingClient):
        self.client = client
        self.output_dir = Path(os.getenv("DEFAULT_OUTPUT_DIR", "./output"))
        self.assessment_output_dir = self.output_dir / "patient_assessments"
        self.assessment_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard assessment templates
        self.assessment_templates = self._load_assessment_templates()
    
    def _load_assessment_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load standard healthcare assessment templates."""
        templates = {
            "mental_health_screening": {
                "name": "Mental Health Screening Assessment",
                "version": "1.0",
                "fields": {
                    "mood_description": {
                        "type": "string",
                        "question": "How would you describe the patient's current mood?",
                        "question_type": "open_ended",
                        "description": "Patient's overall emotional state and mood presentation"
                    },
                    "anxiety_level": {
                        "type": "string",
                        "question": "What is the patient's anxiety level?",
                        "question_type": "single_choice",
                        "enum": ["None", "Mild", "Moderate", "Severe"],
                        "description": "Patient's reported or observed anxiety level"
                    },
                    "sleep_quality": {
                        "type": "string",
                        "question": "How is the patient's sleep quality?",
                        "question_type": "single_choice",
                        "enum": ["Excellent", "Good", "Fair", "Poor", "Very Poor"],
                        "description": "Patient's sleep patterns and quality"
                    },
                    "depression_indicators": {
                        "type": "array",
                        "question": "What depression indicators are present?",
                        "question_type": "multiple_choice",
                        "items": {"type": "string"},
                        "possible_values": [
                            "Persistent sadness",
                            "Loss of interest",
                            "Fatigue",
                            "Appetite changes",
                            "Sleep disturbances",
                            "Difficulty concentrating",
                            "Feelings of worthlessness",
                            "Suicidal thoughts"
                        ],
                        "description": "Observable or reported signs of depression"
                    },
                    "social_support": {
                        "type": "string",
                        "question": "Describe the patient's social support system",
                        "question_type": "open_ended",
                        "description": "Family, friends, and community support available to patient"
                    },
                    "medication_compliance": {
                        "type": "string",
                        "question": "How is the patient's medication compliance?",
                        "question_type": "single_choice",
                        "enum": ["Excellent", "Good", "Fair", "Poor", "Not Applicable"],
                        "description": "Patient's adherence to prescribed medications"
                    },
                    "risk_assessment": {
                        "type": "string",
                        "question": "What is the overall risk level?",
                        "question_type": "single_choice",
                        "enum": ["Low", "Moderate", "High", "Critical"],
                        "description": "Overall assessment of patient risk factors"
                    },
                    "intervention_needed": {
                        "type": "boolean",
                        "question": "Is immediate intervention needed?",
                        "question_type": "yes_no",
                        "description": "Whether immediate clinical intervention is required"
                    },
                    "additional_notes": {
                        "type": "string",
                        "question": "Additional clinical observations or notes",
                        "question_type": "open_ended",
                        "description": "Any additional relevant information from the conversation"
                    }
                }
            },
            
            "physical_health_assessment": {
                "name": "Physical Health Assessment",
                "version": "1.0",
                "fields": {
                    "chief_complaint": {
                        "type": "string",
                        "question": "What is the patient's chief complaint?",
                        "question_type": "open_ended",
                        "description": "Primary reason for the visit or main health concern"
                    },
                    "pain_level": {
                        "type": "integer",
                        "question": "What is the patient's pain level (0-10 scale)?",
                        "question_type": "numeric_scale",
                        "range": [0, 10],
                        "description": "Patient's self-reported pain level on 0-10 scale"
                    },
                    "pain_location": {
                        "type": "array",
                        "question": "Where is the patient experiencing pain?",
                        "question_type": "multiple_choice",
                        "items": {"type": "string"},
                        "possible_values": [
                            "Head", "Neck", "Chest", "Back", "Abdomen", 
                            "Arms", "Legs", "Joints", "Muscles", "Other"
                        ],
                        "description": "Body locations where patient reports pain"
                    },
                    "mobility_status": {
                        "type": "string",
                        "question": "How is the patient's mobility?",
                        "question_type": "single_choice",
                        "enum": ["Independent", "Assisted", "Limited", "Immobile"],
                        "description": "Patient's ability to move and perform daily activities"
                    },
                    "appetite_changes": {
                        "type": "string",
                        "question": "Any changes in appetite?",
                        "question_type": "single_choice",
                        "enum": ["Increased", "Decreased", "No Change", "Loss of Appetite"],
                        "description": "Recent changes in patient's eating patterns"
                    },
                    "symptoms_present": {
                        "type": "array",
                        "question": "What symptoms are present?",
                        "question_type": "multiple_choice",
                        "items": {"type": "string"},
                        "possible_values": [
                            "Fever", "Nausea", "Vomiting", "Dizziness", "Fatigue",
                            "Shortness of breath", "Cough", "Headache", "Rash", "Swelling"
                        ],
                        "description": "Current symptoms reported by patient"
                    },
                    "medication_side_effects": {
                        "type": "string",
                        "question": "Any medication side effects reported?",
                        "question_type": "open_ended",
                        "description": "Side effects from current medications"
                    },
                    "functional_status": {
                        "type": "string",
                        "question": "How is the patient's functional status?",
                        "question_type": "single_choice",
                        "enum": ["Independent", "Partially Dependent", "Dependent", "Bed-bound"],
                        "description": "Patient's ability to perform activities of daily living"
                    }
                }
            },
            
            "geriatric_assessment": {
                "name": "Geriatric Comprehensive Assessment",
                "version": "1.0", 
                "fields": {
                    "cognitive_status": {
                        "type": "string",
                        "question": "How is the patient's cognitive function?",
                        "question_type": "single_choice",
                        "enum": ["Normal", "Mild Impairment", "Moderate Impairment", "Severe Impairment"],
                        "description": "Assessment of patient's cognitive abilities"
                    },
                    "memory_concerns": {
                        "type": "boolean",
                        "question": "Are there memory concerns?",
                        "question_type": "yes_no",
                        "description": "Whether patient or family report memory issues"
                    },
                    "fall_risk": {
                        "type": "string",
                        "question": "What is the patient's fall risk?",
                        "question_type": "single_choice",
                        "enum": ["Low", "Moderate", "High"],
                        "description": "Assessment of patient's risk for falls"
                    },
                    "balance_issues": {
                        "type": "boolean",
                        "question": "Does the patient have balance issues?",
                        "question_type": "yes_no",
                        "description": "Whether patient reports or demonstrates balance problems"
                    },
                    "caregiver_support": {
                        "type": "string",
                        "question": "Describe caregiver support available",
                        "question_type": "open_ended",
                        "description": "Level and type of caregiver support available"
                    },
                    "living_situation": {
                        "type": "string",
                        "question": "What is the patient's living situation?",
                        "question_type": "single_choice",
                        "enum": ["Independent", "Assisted Living", "Nursing Home", "With Family"],
                        "description": "Patient's current living arrangements"
                    },
                    "polypharmacy_concerns": {
                        "type": "boolean",
                        "question": "Are there polypharmacy concerns?",
                        "question_type": "yes_no",
                        "description": "Whether patient is taking multiple medications with potential interactions"
                    },
                    "nutritional_status": {
                        "type": "string",
                        "question": "How is the patient's nutritional status?",
                        "question_type": "single_choice",
                        "enum": ["Well Nourished", "At Risk", "Malnourished"],
                        "description": "Assessment of patient's nutritional health"
                    }
                }
            }
        }
        
        return templates
    
    def create_assessment_analyzer(
        self,
        assessment_type: str,
        analyzer_id: Optional[str] = None,
        force_overwrite: bool = False
    ) -> str:
        """
        Create or reuse a custom analyzer for a specific assessment type.
        
        Args:
            assessment_type: Type of assessment (mental_health_screening, physical_health_assessment, etc.)
            analyzer_id: Optional custom analyzer ID
            force_overwrite: If True, delete and recreate existing analyzer
            
        Returns:
            Analyzer ID (existing or newly created)
        """
        try:
            if assessment_type not in self.assessment_templates:
                raise ValueError(f"Unknown assessment type: {assessment_type}")
            
            template = self.assessment_templates[assessment_type]
            
            if analyzer_id is None:
                analyzer_id = f"healthcare-{assessment_type.replace('_', '-')}-analyzer"
            
            # Check if analyzer already exists
            logger.info("Getting list of existing analyzers...")
            try:
                existing_analyzer_ids = self.client.list_analyzer_ids()
                logger.info(f"Raw existing_analyzers response: {existing_analyzer_ids}")
                
                if analyzer_id in existing_analyzer_ids:
                    if not force_overwrite:
                        logger.info(f"Reusing existing analyzer: {analyzer_id}")
                        return analyzer_id
                    else:
                        logger.info(f"Deleting existing analyzer for recreation: {analyzer_id}")
                        self.client.delete_analyzer(analyzer_id)
                        
            except Exception as list_error:
                logger.error(f"Error listing analyzers: {list_error}")
                # Try to delete anyway in case it exists
                try:
                    logger.info(f"Attempting to delete analyzer {analyzer_id} due to list error...")
                    self.client.delete_analyzer(analyzer_id)
                    logger.info(f"Successfully deleted analyzer: {analyzer_id}")
                except Exception as delete_error:
                    logger.info(f"Could not delete analyzer {analyzer_id}: {delete_error}")
            
        except Exception as e:
            import traceback
            logger.error(f"Error in create_assessment_analyzer: {traceback.format_exc()}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            raise
        
        analyzer_config = {
            "description": f"Healthcare analyzer for {template['name']}",
            "baseAnalyzerId": "prebuilt-callCenter",  # Use call center for healthcare conversations
            "config": {
                "locales": ["en-US"],
                "returnDetails": True,
                "disableContentFiltering": False  # Important for healthcare accuracy
            },
            "fieldSchema": {
                "fields": template["fields"]
            }
        }
        
        logger.info(f"Creating analyzer: {analyzer_id}")
        result = self.client.create_analyzer(
            analyzer_id=analyzer_id,
            analyzer_config=analyzer_config
        )
        
        logger.info(f"Analyzer created successfully: {analyzer_id}")
        return analyzer_id
    
    def analyze_patient_conversation(
        self,
        audio_url: str,
        assessment_type: str,
        patient_id: Optional[str] = None,
        care_staff_id: Optional[str] = None,
        session_metadata: Optional[Dict[str, Any]] = None,
        analyzer_id: Optional[str] = None,
        force_overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze a patient-care staff conversation and extract assessment data.
        
        Args:
            audio_url: URL of the conversation audio
            assessment_type: Type of assessment to perform
            patient_id: Optional patient identifier
            care_staff_id: Optional care staff identifier  
            session_metadata: Optional session metadata
            analyzer_id: Optional custom analyzer ID
            force_overwrite: If True, delete and recreate existing analyzer
            
        Returns:
            Complete assessment results
        """
        # Create analyzer for this assessment type
        analyzer_id = self.create_assessment_analyzer(
            assessment_type,
            analyzer_id=analyzer_id,
            force_overwrite=force_overwrite
        )
        
        # Analyze the conversation
        logger.info(f"Starting analysis of patient conversation for {assessment_type}")
        try:
            analysis_request = self.client.analyze_content(
                analyzer_id=analyzer_id,
                content_url=audio_url
            )
            
            request_id = analysis_request["request_id"]
            if not request_id:
                raise ValueError(f"No request ID found in response: {analysis_request}")

            logger.info(f"Analysis started with request ID: {request_id}")
            
            # Wait for completion (healthcare analysis may take longer for accuracy)
            result = self.client.wait_for_analysis_completion(
                request_id,
                max_wait_time=600,  # 10 minutes
                poll_interval=10
            )
            
        except Exception as e:
            import traceback
            logger.error(f"Detailed error: {traceback.format_exc()}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            raise
        
        # Process results into assessment format
        assessment_result = self._process_assessment_results(
            result,
            assessment_type,
            patient_id,
            care_staff_id,
            session_metadata
        )
        
        return assessment_result
    
    def _process_assessment_results(
        self,
        raw_result: Dict[str, Any],
        assessment_type: str,
        patient_id: Optional[str],
        care_staff_id: Optional[str],
        session_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process raw analysis results into structured assessment format."""
        template = self.assessment_templates[assessment_type]
        
        # Extract conversation transcript
        transcript = extract_transcript_text(raw_result)
        
        # Extract analysis results
        contents = raw_result.get("result", {}).get("contents", [])
        extracted_fields = {}
        extraction_status = {}
        
        if contents:
            content = contents[0]
            fields = content.get("fields", {})
            
            for field_name, field_data in fields.items():
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
                
                # Store the extracted value
                extracted_fields[field_name] = extracted_value
                
                # Set extraction status based on whether we extracted a value
                # 1.0 if extracted, 0.0 if missing/empty
                if extracted_value is not None and extracted_value != "" and extracted_value != []:
                    extraction_status[field_name] = 1.0
                else:
                    extraction_status[field_name] = 0.0
        
        # Create comprehensive assessment result
        assessment_result = {
            "assessment_metadata": {
                "assessment_type": assessment_type,
                "assessment_name": template["name"],
                "template_version": template["version"],
                "analysis_timestamp": datetime.now().isoformat(),
                "patient_id": patient_id,
                "care_staff_id": care_staff_id,
                "session_metadata": session_metadata or {}
            },
            "conversation_data": {
                "transcript": transcript,
                "duration_info": self._extract_duration_info(contents[0] if contents else {}),
                "speaker_info": self._extract_speaker_info(contents[0] if contents else {})
            },
            "assessment_responses": {},
            "extraction_status": extraction_status,
            "quality_indicators": self._calculate_quality_indicators(extraction_status),
            "recommendations": self._generate_recommendations(extracted_fields, extraction_status, assessment_type),
            "raw_analysis_result": raw_result
        }
        
        # Map extracted fields to assessment questions
        for field_name, field_config in template["fields"].items():
            question_type = field_config["question_type"]
            question_text = field_config["question"]
            extracted_value = extracted_fields.get(field_name)
            field_extracted = extraction_status.get(field_name, 0.0)
            
            assessment_response = {
                "question": question_text,
                "question_type": question_type,
                "extracted_value": extracted_value,
                "field_extracted": field_extracted,
                "requires_review": extracted_value is None or extracted_value == "",  # Flag missing values
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
        
        return assessment_result
    
    def _extract_duration_info(self, content: Dict[str, Any]) -> Dict[str, Any]:
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
    
    def _extract_speaker_info(self, content: Dict[str, Any]) -> Dict[str, Any]:
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
                "start_time_ms": phrase.get("offsetMs", 0),
                "duration_ms": phrase.get("durationMs", 0)
            })
        
        speaker_info["speakers_detected"] = list(speakers)
        speaker_info["speaker_count"] = len(speakers)
        
        return speaker_info
    
    def _calculate_quality_indicators(self, extraction_status: Dict[str, float]) -> Dict[str, Any]:
        """Calculate quality indicators for the assessment (adapted for audio processing)."""
        if not extraction_status:
            return {"overall_quality": "insufficient_data"}
        
        # For audio processing, we treat extracted values as high quality
        # extraction_status contains 1.0 for extracted fields, 0.0 for missing
        total_fields = len(extraction_status)
        extracted_fields = sum(1 for status in extraction_status.values() if status > 0)
        missing_fields = total_fields - extracted_fields
        extraction_rate = extracted_fields / total_fields if total_fields > 0 else 0
        
        quality_indicators = {
            "extraction_rate": extraction_rate,
            "extracted_fields": extracted_fields,
            "missing_fields": missing_fields,
            "total_fields": total_fields,
            "overall_quality": "high" if extraction_rate >= 0.8 else "medium" if extraction_rate >= 0.5 else "low"
        }
        
        return quality_indicators
    
    def _generate_recommendations(
        self,
        extracted_fields: Dict[str, Any],
        extraction_status: Dict[str, float],
        assessment_type: str
    ) -> List[str]:
        """Generate recommendations based on assessment results (adapted for audio processing)."""
        recommendations = []
        
        # Check for missing fields
        missing_fields = [
            field for field, status in extraction_status.items() if status == 0 or extracted_fields.get(field) is None
        ]
        
        if missing_fields:
            recommendations.append(
                f"Review audio recording for the following missing information: {', '.join(missing_fields)}"
            )
        
        # Assessment-specific recommendations
        if assessment_type == "mental_health_screening":
            if extracted_fields.get("intervention_needed"):
                recommendations.append("URGENT: Immediate clinical intervention may be required")
            
            if extracted_fields.get("risk_assessment") in ["High", "Critical"]:
                recommendations.append("High risk patient - ensure appropriate follow-up care")
        
        elif assessment_type == "physical_health_assessment":
            pain_level = extracted_fields.get("pain_level")
            if isinstance(pain_level, int) and pain_level >= 7:
                recommendations.append("High pain level reported - consider pain management intervention")
        
        elif assessment_type == "geriatric_assessment":
            if extracted_fields.get("fall_risk") == "High":
                recommendations.append("High fall risk - implement fall prevention measures")
            
            if extracted_fields.get("cognitive_status") in ["Moderate Impairment", "Severe Impairment"]:
                recommendations.append("Cognitive impairment noted - consider cognitive assessment referral")
        
        # Data quality recommendations
        if not recommendations:
            recommendations.append("Assessment completed successfully - all fields extracted successfully")
        
        return recommendations
    
    def save_assessment_result(
        self,
        assessment_result: Dict[str, Any],
        filename: Optional[str] = None
    ) -> Path:
        """Save assessment result to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            assessment_type = assessment_result["assessment_metadata"]["assessment_type"]
            patient_id = assessment_result["assessment_metadata"].get("patient_id", "unknown")
            filename = f"{assessment_type}_{patient_id}_{timestamp}.json"
        
        output_path = self.assessment_output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(assessment_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Assessment result saved to: {output_path}")
        return output_path
    
    def generate_assessment_report(
        self,
        assessment_result: Dict[str, Any],
        format_type: str = "summary"
    ) -> str:
        """Generate a human-readable assessment report."""
        metadata = assessment_result["assessment_metadata"]
        responses = assessment_result["assessment_responses"]
        quality = assessment_result["quality_indicators"]
        recommendations = assessment_result["recommendations"]
        
        report = f"""
HEALTHCARE PATIENT ASSESSMENT REPORT
=====================================

Assessment Information:
- Type: {metadata['assessment_name']}
- Date/Time: {metadata['analysis_timestamp']}
- Patient ID: {metadata.get('patient_id', 'Not specified')}
- Care Staff ID: {metadata.get('care_staff_id', 'Not specified')}

Quality Indicators:
- Overall Quality: {quality['overall_quality'].upper()}
- Extraction Rate: {quality['extraction_rate']:.2f}
- Fields Extracted: {quality['extracted_fields']}/{quality['total_fields']}
- Missing Fields: {quality['missing_fields']}

Assessment Responses:
"""
        
        for field_name, response in responses.items():
            status = "⚠️" if response["requires_review"] else "✅"
            field_extracted = response["field_extracted"]
            value = response["extracted_value"]
            
            if isinstance(value, list):
                value_str = ", ".join(map(str, value)) if value else "None"
            else:
                value_str = str(value) if value is not None else "Not answered"
            
            extraction_status = "Extracted" if field_extracted > 0 else "Missing"
            
            report += f"""
{status} {response['question']}
   Answer: {value_str}
   Status: {extraction_status}
"""
        
        report += f"""
Recommendations:
"""
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += f"""
---
This assessment was generated automatically from conversation audio.
Please review all flagged items and verify accuracy before clinical use.
"""
        
        return report


def create_sample_healthcare_scenarios() -> List[Dict[str, Any]]:
    """Create sample healthcare conversation scenarios for testing."""
    scenarios = [
        {
            "name": "Mental Health Check-in",
            "assessment_type": "mental_health_screening",
            "description": "Routine mental health screening conversation",
            "sample_conversation_topics": [
                "How have you been feeling lately?",
                "Are you sleeping well?",
                "Any changes in appetite or energy?",
                "How are your relationships and social connections?",
                "Are you taking your medications as prescribed?",
                "Any thoughts of self-harm?",
                "What support systems do you have?"
            ],
            "patient_profile": {
                "age_group": "Adult",
                "condition": "Depression follow-up",
                "risk_level": "Moderate"
            }
        },
        {
            "name": "Physical Health Assessment", 
            "assessment_type": "physical_health_assessment",
            "description": "General physical health evaluation",
            "sample_conversation_topics": [
                "What brings you in today?",
                "On a scale of 1-10, how would you rate your pain?",
                "Where specifically are you experiencing discomfort?",
                "How has your mobility been?",
                "Any changes in appetite or weight?",
                "Are you experiencing any new symptoms?",
                "How are you managing your current medications?"
            ],
            "patient_profile": {
                "age_group": "Adult",
                "condition": "Chronic pain management",
                "mobility": "Limited"
            }
        },
        {
            "name": "Geriatric Assessment",
            "assessment_type": "geriatric_assessment", 
            "description": "Comprehensive geriatric evaluation",
            "sample_conversation_topics": [
                "How is your memory lately?",
                "Any concerns about balance or falling?",
                "Tell me about your living situation",
                "Who helps you with daily activities?",
                "How many medications are you currently taking?",
                "How is your appetite and nutrition?",
                "Any confusion or disorientation?"
            ],
            "patient_profile": {
                "age_group": "Senior (75+)",
                "condition": "Multiple chronic conditions",
                "living_situation": "Assisted living"
            }
        }
    ]
    
    return scenarios


def main():
    """Main demo function for healthcare patient assessment."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Healthcare Patient Assessment Demo using Azure AI Content Understanding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with URL
  python healthcare_assessment_demo.py --audio-url "https://example.com/patient_conversation.wav"

  # With patient/staff IDs and specific assessment
  python healthcare_assessment_demo.py --audio-url "https://example.com/audio.wav" --assessment-type "physical_health_assessment" --patient-id "PATIENT_123" --staff-id "NURSE_456"
  
  # Overwrite existing analyzer
  python healthcare_assessment_demo.py --audio-url "https://example.com/audio.wav" --force-overwrite
  
  # Custom analyzer ID
  python healthcare_assessment_demo.py --audio-url "https://example.com/audio.wav" --analyzer-id "my-custom-analyzer"
        """
    )
    
    parser.add_argument(
        "--audio-url",
        type=str,
        help="URL of the audio file containing patient-care staff conversation"
    )
    
    parser.add_argument(
        "--assessment-type",
        type=str,
        choices=["mental_health_screening", "physical_health_assessment", "geriatric_assessment"],
        default="mental_health_screening",
        help="Type of healthcare assessment to perform (default: mental_health_screening)"
    )
    
    parser.add_argument(
        "--patient-id",
        type=str,
        help="Patient identifier for the assessment"
    )
    
    parser.add_argument(
        "--staff-id",
        type=str,
        help="Care staff identifier for the assessment"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save assessment results (overrides DEFAULT_OUTPUT_DIR)"
    )
    
    parser.add_argument(
        "--list-assessments",
        action="store_true",
        help="List available assessment types and exit"
    )
    
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Delete and recreate existing analyzers (use with caution)"
    )
    
    parser.add_argument(
        "--analyzer-id",
        type=str,
        help="Custom analyzer ID to use (optional)"
    )
    
    args = parser.parse_args()
    
    load_dotenv()
    
    print("🏥 Healthcare Patient Assessment Demo")
    print("Azure AI Content Understanding for Clinical Conversations")
    print("=" * 60)
    
    try:
        # Create client and analyzer
        client = create_client_from_env()
        assessment_analyzer = PatientAssessmentAnalyzer(client)
        
        # Override output directory if specified
        if args.output_dir:
            assessment_analyzer.output_dir = Path(args.output_dir)
            assessment_analyzer.assessment_output_dir = assessment_analyzer.output_dir / "patient_assessments"
            assessment_analyzer.assessment_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Display available assessment types
        print("\nAvailable Assessment Types:")
        for assessment_type, template in assessment_analyzer.assessment_templates.items():
            marker = "→" if assessment_type == args.assessment_type else " "
            print(f"  {marker} {assessment_type}: {template['name']}")
        
        if args.list_assessments:
            print("\nUse --assessment-type to specify which assessment to run")
            return
        
        # Check if audio URL is provided
        if not args.audio_url:
            print("\n❌ Error: Audio URL is required")
            print("Use --audio-url to specify the audio file URL")
            print("Examples:")
            print("  python healthcare_assessment_demo.py --audio-url 'https://example.com/conversation.wav'")
            sys.exit(1)
        
        print(f"\n=== {args.assessment_type.replace('_', ' ').title()} Demo ===")
        print(f"Audio URL: {args.audio_url}")
        print(f"Patient ID: {args.patient_id or 'Not specified'}")
        print(f"Staff ID: {args.staff_id or 'Not specified'}")
        
        # Display analyzer behavior
        if args.force_overwrite:
            analyzer_behavior = "Force overwrite existing analyzers"
        else:
            analyzer_behavior = "Reuse existing analyzers"
        print(f"Analyzer Mode: {analyzer_behavior}")
        
        # Prepare session metadata
        session_metadata = {
            "session_type": "ai_analysis_demo",
            "audio_source": args.audio_url,
            "audio_source_type": "URL",
            "analysis_tool": "azure_ai_content_understanding",
            "analyzer_behavior": analyzer_behavior
        }
        
        # Analyze conversation for specified assessment type
        print("\n🔄 Analyzing patient conversation...")
        assessment_result = assessment_analyzer.analyze_patient_conversation(
            audio_url=args.audio_url,
            assessment_type=args.assessment_type,
            patient_id=args.patient_id,
            care_staff_id=args.staff_id,
            session_metadata=session_metadata,
            analyzer_id=args.analyzer_id,
            force_overwrite=args.force_overwrite
        )
        
        # Save results
        output_path = assessment_analyzer.save_assessment_result(assessment_result)
        
        # Generate and display report
        report = assessment_analyzer.generate_assessment_report(assessment_result)
        print(report)
        
        # Save report
        report_path = output_path.parent / f"report_{output_path.stem}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ Assessment completed successfully!")
        print(f"📁 Results saved to: {output_path}")
        print(f"📄 Report saved to: {report_path}")
        
        # Display quality summary
        quality = assessment_result["quality_indicators"]
        print(f"\n📊 Quality Summary:")
        print(f"   Overall Quality: {quality['overall_quality'].upper()}")
        print(f"   Extraction Rate: {quality['extraction_rate']:.2f}")
        print(f"   Fields Extracted: {quality['extracted_fields']}/{quality['total_fields']}")
        print(f"   Missing Fields: {quality['missing_fields']}")
        
        # Display recommendations
        recommendations = assessment_result["recommendations"]
        if recommendations:
            print(f"\n🎯 Key Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                print(f"   {i}. {rec}")
        
        print(f"\n💡 Usage Tips:")
        print("1. Always review AI-generated assessments before clinical use")
        print("2. Pay attention to extraction status for each field")
        print("3. Manually verify any fields marked for review")
        print("4. Ensure HIPAA compliance when handling patient data")
        
    except Exception as e:
        print(f"❌ Healthcare assessment demo failed: {e}")
        logger.error("Healthcare assessment demo failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
