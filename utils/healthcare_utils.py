"""
Healthcare Utilities for Azure AI Content Understanding

This module provides healthcare-specific utilities including HIPAA compliance helpers,
clinical data validation, and assessment form management.
"""

import json
import hashlib
from typing import Dict, List, Any
from datetime import datetime, timedelta
from pathlib import Path
import re


class HIPAAComplianceHelper:
    """
    Helper class for HIPAA compliance when processing healthcare conversations.
    Provides utilities for data anonymization, audit logging, and secure handling.
    """
    
    def __init__(self, enable_audit_logging: bool = True):
        self.enable_audit_logging = enable_audit_logging
        self.audit_log_path = Path("./output/audit_log.json")
        self.audit_log_path.parent.mkdir(exist_ok=True)
    
    def anonymize_patient_data(
        self,
        assessment_result: Dict[str, Any],
        keep_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Anonymize patient data in assessment results for research or training.
        
        Args:
            assessment_result: Original assessment results
            keep_metadata: Whether to keep non-identifying metadata
            
        Returns:
            Anonymized assessment results
        """
        anonymized = assessment_result.copy()
        
        # Replace patient identifiers
        metadata = anonymized.get("assessment_metadata", {})
        
        if "patient_id" in metadata:
            # Create anonymized patient ID hash
            original_id = metadata["patient_id"]
            anonymized_id = self._create_anonymous_id(original_id, "PATIENT")
            metadata["patient_id"] = anonymized_id
            
            # Log the anonymization
            self._log_anonymization_event(original_id, anonymized_id, "patient")
        
        if "care_staff_id" in metadata:
            original_staff_id = metadata["care_staff_id"]
            anonymized_staff_id = self._create_anonymous_id(original_staff_id, "STAFF")
            metadata["care_staff_id"] = anonymized_staff_id
            
            self._log_anonymization_event(original_staff_id, anonymized_staff_id, "staff")
        
        # Remove or anonymize conversation transcript
        if "conversation_data" in anonymized:
            conversation_data = anonymized["conversation_data"]
            
            # Replace transcript with anonymized version
            if "transcript" in conversation_data:
                original_transcript = conversation_data["transcript"]
                anonymized_transcript = self._anonymize_transcript(original_transcript)
                conversation_data["transcript"] = anonymized_transcript
            
            # Anonymize speaker information
            if "speaker_info" in conversation_data:
                speaker_info = conversation_data["speaker_info"]
                if "speaker_segments" in speaker_info:
                    for segment in speaker_info["speaker_segments"]:
                        if "text" in segment:
                            segment["text"] = self._anonymize_transcript(segment["text"])
        
        # Add anonymization metadata
        anonymized["anonymization_info"] = {
            "anonymized": True,
            "anonymization_timestamp": datetime.now().isoformat(),
            "anonymization_method": "hash_based",
            "original_identifiers_removed": True
        }
        
        return anonymized
    
    def _create_anonymous_id(self, original_id: str, prefix: str) -> str:
        """Create a consistent anonymous ID from original ID."""
        # Use SHA-256 hash for consistent anonymization
        hash_object = hashlib.sha256(original_id.encode())
        hash_hex = hash_object.hexdigest()
        return f"{prefix}_{hash_hex[:8].upper()}"
    
    def _anonymize_transcript(self, transcript: str) -> str:
        """
        Anonymize personally identifiable information in transcript text.
        
        Args:
            transcript: Original transcript text
            
        Returns:
            Anonymized transcript text
        """
        anonymized = transcript
        
        # Define patterns for PII
        pii_patterns = {
            r'\b\d{3}-\d{2}-\d{4}\b': '[SSN]',  # Social Security Numbers
            r'\b\d{3}-\d{3}-\d{4}\b': '[PHONE]',  # Phone Numbers
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]',  # Email
            r'\b\d{1,5}\s+\w+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b': '[ADDRESS]',  # Addresses
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b': '[DATE]',  # Dates
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b': '[DATE]',  # Date formats
        }
        
        # Apply anonymization patterns
        for pattern, replacement in pii_patterns.items():
            anonymized = re.sub(pattern, replacement, anonymized, flags=re.IGNORECASE)
        
        # Anonymize common names (basic approach)
        # In production, you would use a more sophisticated NER model
        common_names = [
            'John', 'Jane', 'Michael', 'Sarah', 'David', 'Lisa', 'Robert', 'Mary',
            'James', 'Patricia', 'Christopher', 'Jennifer', 'Daniel', 'Linda'
        ]
        
        for name in common_names:
            pattern = r'\b' + re.escape(name) + r'\b'
            anonymized = re.sub(pattern, '[NAME]', anonymized, flags=re.IGNORECASE)
        
        return anonymized
    
    def _log_anonymization_event(
        self,
        original_id: str,
        anonymized_id: str,
        id_type: str
    ):
        """Log anonymization events for audit purposes."""
        if not self.enable_audit_logging:
            return
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "anonymization",
            "id_type": id_type,
            "original_id_hash": hashlib.sha256(original_id.encode()).hexdigest(),
            "anonymized_id": anonymized_id,
            "action": "identifier_anonymized"
        }
        
        self._write_audit_log(event)
    
    def _write_audit_log(self, event: Dict[str, Any]):
        """Write event to audit log."""
        try:
            # Read existing log
            if self.audit_log_path.exists():
                with open(self.audit_log_path, 'r') as f:
                    audit_log = json.load(f)
            else:
                audit_log = {"events": []}
            
            # Add new event
            audit_log["events"].append(event)
            
            # Write back to file
            with open(self.audit_log_path, 'w') as f:
                json.dump(audit_log, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not write to audit log: {e}")
    
    def validate_data_retention_policy(
        self,
        assessment_result: Dict[str, Any],
        retention_days: int = 2555  # 7 years default for healthcare
    ) -> Dict[str, Any]:
        """
        Validate and apply data retention policies.
        
        Args:
            assessment_result: Assessment results to validate
            retention_days: Number of days to retain data
            
        Returns:
            Validation results and retention information
        """
        metadata = assessment_result.get("assessment_metadata", {})
        created_timestamp = metadata.get("analysis_timestamp")
        
        if not created_timestamp:
            return {
                "valid": False,
                "error": "No creation timestamp found",
                "action_required": "Add timestamp to assessment metadata"
            }
        
        try:
            created_date = datetime.fromisoformat(created_timestamp.replace('Z', '+00:00'))
            expiry_date = created_date + timedelta(days=retention_days)
            current_date = datetime.now()
            
            days_remaining = (expiry_date - current_date).days
            
            retention_info = {
                "valid": True,
                "created_date": created_date.isoformat(),
                "expiry_date": expiry_date.isoformat(),
                "days_remaining": days_remaining,
                "retention_period_days": retention_days,
                "status": "active" if days_remaining > 0 else "expired"
            }
            
            if days_remaining <= 30:
                retention_info["warning"] = "Data approaching retention expiry"
                retention_info["action_required"] = "Review for archival or deletion"
            
            return retention_info
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Invalid timestamp format: {e}",
                "action_required": "Fix timestamp format in assessment metadata"
            }


class ClinicalDataValidator:
    """
    Validates clinical assessment data for completeness, consistency, and accuracy.
    """
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load clinical validation rules."""
        return {
            "mental_health_screening": {
                "required_fields": [
                    "mood_description", "anxiety_level", "sleep_quality", 
                    "risk_assessment", "intervention_needed"
                ],
                "consistency_checks": [
                    {
                        "name": "high_risk_intervention_check",
                        "condition": "risk_assessment in ['High', 'Critical']",
                        "expected": "intervention_needed == True",
                        "message": "High/Critical risk should typically require intervention"
                    }
                ],
                "value_ranges": {
                    "anxiety_level": ["None", "Mild", "Moderate", "Severe"],
                    "sleep_quality": ["Excellent", "Good", "Fair", "Poor", "Very Poor"],
                    "risk_assessment": ["Low", "Moderate", "High", "Critical"]
                }
            },
            "physical_health_assessment": {
                "required_fields": [
                    "chief_complaint", "pain_level", "mobility_status", "functional_status"
                ],
                "consistency_checks": [
                    {
                        "name": "pain_mobility_check",
                        "condition": "pain_level >= 7",
                        "expected": "mobility_status in ['Limited', 'Assisted', 'Immobile']",
                        "message": "High pain levels typically affect mobility"
                    }
                ],
                "value_ranges": {
                    "pain_level": list(range(0, 11)),  # 0-10 scale
                    "mobility_status": ["Independent", "Assisted", "Limited", "Immobile"],
                    "functional_status": ["Independent", "Partially Dependent", "Dependent", "Bed-bound"]
                }
            },
            "geriatric_assessment": {
                "required_fields": [
                    "cognitive_status", "fall_risk", "living_situation", "caregiver_support"
                ],
                "consistency_checks": [
                    {
                        "name": "cognitive_living_check",
                        "condition": "cognitive_status in ['Moderate Impairment', 'Severe Impairment']",
                        "expected": "living_situation in ['Assisted Living', 'Nursing Home', 'With Family']",
                        "message": "Cognitive impairment typically requires supervised living"
                    }
                ],
                "value_ranges": {
                    "cognitive_status": ["Normal", "Mild Impairment", "Moderate Impairment", "Severe Impairment"],
                    "fall_risk": ["Low", "Moderate", "High"],
                    "living_situation": ["Independent", "Assisted Living", "Nursing Home", "With Family"]
                }
            }
        }
    
    def validate_assessment(
        self,
        assessment_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive validation of assessment results.
        
        Args:
            assessment_result: Assessment results to validate
            
        Returns:
            Validation report with findings and recommendations
        """
        assessment_type = assessment_result.get("assessment_metadata", {}).get("assessment_type")
        
        if assessment_type not in self.validation_rules:
            return {
                "valid": False,
                "error": f"Unknown assessment type: {assessment_type}",
                "validation_results": {}
            }
        
        rules = self.validation_rules[assessment_type]
        responses = assessment_result.get("assessment_responses", {})
        
        validation_results = {
            "completeness": self._check_completeness(responses, rules),
            "consistency": self._check_consistency(responses, rules),
            "value_validity": self._check_value_validity(responses, rules),
            "confidence_analysis": self._analyze_confidence(assessment_result),
            "clinical_flags": self._check_clinical_flags(responses, assessment_type)
        }
        
        # Overall validation status
        overall_valid = all([
            validation_results["completeness"]["passed"],
            validation_results["consistency"]["passed"],
            validation_results["value_validity"]["passed"]
        ])
        
        return {
            "valid": overall_valid,
            "assessment_type": assessment_type,
            "validation_timestamp": datetime.now().isoformat(),
            "validation_results": validation_results,
            "recommendations": self._generate_validation_recommendations(validation_results)
        }
    
    def _check_completeness(
        self,
        responses: Dict[str, Any],
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if all required fields are present and answered."""
        required_fields = rules.get("required_fields", [])
        missing_fields = []
        unanswered_fields = []
        
        for field in required_fields:
            if field not in responses:
                missing_fields.append(field)
            else:
                response = responses[field]
                extracted_value = response.get("extracted_value")
                
                if extracted_value is None or extracted_value == "" or extracted_value == []:
                    unanswered_fields.append(field)
        
        return {
            "passed": len(missing_fields) == 0 and len(unanswered_fields) == 0,
            "missing_fields": missing_fields,
            "unanswered_fields": unanswered_fields,
            "completion_rate": (len(required_fields) - len(missing_fields) - len(unanswered_fields)) / len(required_fields)
        }
    
    def _check_consistency(
        self,
        responses: Dict[str, Any],
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check logical consistency between related fields."""
        consistency_checks = rules.get("consistency_checks", [])
        failed_checks = []
        
        for check in consistency_checks:
            try:
                # Create a context with current response values
                context = {}
                for field_name, response in responses.items():
                    context[field_name] = response.get("extracted_value")
                
                # Evaluate condition and expected result
                condition_met = self._evaluate_condition(check["condition"], context)
                expected_met = self._evaluate_condition(check["expected"], context)
                
                if condition_met and not expected_met:
                    failed_checks.append({
                        "check_name": check["name"],
                        "message": check["message"],
                        "condition": check["condition"],
                        "expected": check["expected"]
                    })
                    
            except Exception as e:
                # Skip checks that can't be evaluated
                continue
        
        return {
            "passed": len(failed_checks) == 0,
            "failed_checks": failed_checks,
            "total_checks": len(consistency_checks)
        }
    
    def _check_value_validity(
        self,
        responses: Dict[str, Any],
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if extracted values are within valid ranges."""
        value_ranges = rules.get("value_ranges", {})
        invalid_values = []
        
        for field_name, valid_range in value_ranges.items():
            if field_name in responses:
                extracted_value = responses[field_name].get("extracted_value")
                
                if extracted_value is not None and extracted_value not in valid_range:
                    invalid_values.append({
                        "field": field_name,
                        "extracted_value": extracted_value,
                        "valid_range": valid_range
                    })
        
        return {
            "passed": len(invalid_values) == 0,
            "invalid_values": invalid_values,
            "total_validated_fields": len(value_ranges)
        }
    
    def _analyze_confidence(self, assessment_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze confidence scores across the assessment."""
        confidence_scores = assessment_result.get("confidence_scores", {})
        
        if not confidence_scores:
            return {"analysis_available": False}
        
        scores = list(confidence_scores.values())
        low_confidence_threshold = 0.7
        very_low_confidence_threshold = 0.5
        
        low_confidence_fields = [
            field for field, score in confidence_scores.items() 
            if score < low_confidence_threshold
        ]
        
        very_low_confidence_fields = [
            field for field, score in confidence_scores.items()
            if score < very_low_confidence_threshold
        ]
        
        return {
            "analysis_available": True,
            "average_confidence": sum(scores) / len(scores),
            "minimum_confidence": min(scores),
            "maximum_confidence": max(scores),
            "low_confidence_fields": low_confidence_fields,
            "very_low_confidence_fields": very_low_confidence_fields,
            "confidence_distribution": self._calculate_confidence_distribution(scores)
        }
    
    def _check_clinical_flags(
        self,
        responses: Dict[str, Any],
        assessment_type: str
    ) -> List[Dict[str, Any]]:
        """Check for clinically significant findings that require attention."""
        flags = []
        
        if assessment_type == "mental_health_screening":
            # Check for high-risk indicators
            intervention_needed = responses.get("intervention_needed", {}).get("extracted_value")
            if intervention_needed:
                flags.append({
                    "type": "urgent",
                    "field": "intervention_needed",
                    "message": "Immediate intervention flagged",
                    "action": "Contact supervising clinician immediately"
                })
            
            risk_level = responses.get("risk_assessment", {}).get("extracted_value")
            if risk_level in ["High", "Critical"]:
                flags.append({
                    "type": "high_risk",
                    "field": "risk_assessment",
                    "message": f"{risk_level} risk level identified",
                    "action": "Implement appropriate safety measures"
                })
        
        elif assessment_type == "physical_health_assessment":
            # Check for severe pain
            pain_level = responses.get("pain_level", {}).get("extracted_value")
            if isinstance(pain_level, int) and pain_level >= 8:
                flags.append({
                    "type": "severe_pain",
                    "field": "pain_level",
                    "message": f"Severe pain reported (level {pain_level})",
                    "action": "Consider immediate pain management intervention"
                })
        
        elif assessment_type == "geriatric_assessment":
            # Check for high fall risk
            fall_risk = responses.get("fall_risk", {}).get("extracted_value")
            if fall_risk == "High":
                flags.append({
                    "type": "fall_risk",
                    "field": "fall_risk",
                    "message": "High fall risk identified",
                    "action": "Implement fall prevention measures"
                })
        
        return flags
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Safely evaluate a condition string with given context."""
        try:
            # Replace field names with their values in the condition
            for field_name, value in context.items():
                if isinstance(value, str):
                    condition = condition.replace(field_name, f"'{value}'")
                else:
                    condition = condition.replace(field_name, str(value))
            
            # Evaluate the condition (be careful about security in production)
            return eval(condition)
        except:
            return False
    
    def _calculate_confidence_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calculate distribution of confidence scores."""
        distribution = {
            "high (>= 0.8)": 0,
            "medium (0.6-0.8)": 0,
            "low (< 0.6)": 0
        }
        
        for score in scores:
            if score >= 0.8:
                distribution["high (>= 0.8)"] += 1
            elif score >= 0.6:
                distribution["medium (0.6-0.8)"] += 1
            else:
                distribution["low (< 0.6)"] += 1
        
        return distribution
    
    def _generate_validation_recommendations(
        self,
        validation_results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Completeness recommendations
        completeness = validation_results["completeness"]
        if not completeness["passed"]:
            if completeness["missing_fields"]:
                recommendations.append(
                    f"Complete missing required fields: {', '.join(completeness['missing_fields'])}"
                )
            if completeness["unanswered_fields"]:
                recommendations.append(
                    f"Obtain answers for unanswered fields: {', '.join(completeness['unanswered_fields'])}"
                )
        
        # Consistency recommendations
        consistency = validation_results["consistency"]
        if not consistency["passed"]:
            for failed_check in consistency["failed_checks"]:
                recommendations.append(f"Review consistency: {failed_check['message']}")
        
        # Value validity recommendations
        value_validity = validation_results["value_validity"]
        if not value_validity["passed"]:
            for invalid_value in value_validity["invalid_values"]:
                recommendations.append(
                    f"Verify value for {invalid_value['field']}: '{invalid_value['extracted_value']}' "
                    f"is not in valid range {invalid_value['valid_range']}"
                )
        
        # Confidence recommendations
        confidence = validation_results.get("confidence_analysis", {})
        if confidence.get("analysis_available"):
            if confidence["very_low_confidence_fields"]:
                recommendations.append(
                    f"Manually verify fields with very low confidence: "
                    f"{', '.join(confidence['very_low_confidence_fields'])}"
                )
        
        # Clinical flags recommendations
        clinical_flags = validation_results.get("clinical_flags", [])
        for flag in clinical_flags:
            recommendations.append(f"CLINICAL ALERT: {flag['message']} - {flag['action']}")
        
        if not recommendations:
            recommendations.append("Assessment validation passed - ready for clinical review")
        
        return recommendations


def create_assessment_template_library() -> Dict[str, Any]:
    """Create a library of assessment templates for different healthcare specialties."""
    return {
        "cardiology_assessment": {
            "name": "Cardiovascular Health Assessment",
            "specialty": "Cardiology",
            "fields": {
                "chest_pain": {
                    "type": "string",
                    "question": "Is the patient experiencing chest pain?",
                    "question_type": "single_choice",
                    "enum": ["None", "Mild", "Moderate", "Severe"],
                    "description": "Current chest pain level"
                },
                "shortness_of_breath": {
                    "type": "boolean",
                    "question": "Is the patient experiencing shortness of breath?",
                    "question_type": "yes_no",
                    "description": "Presence of dyspnea"
                },
                "heart_rate_irregular": {
                    "type": "boolean",
                    "question": "Does the patient report irregular heartbeat?",
                    "question_type": "yes_no",
                    "description": "Irregularities in heart rhythm"
                },
                "exercise_tolerance": {
                    "type": "string",
                    "question": "How is the patient's exercise tolerance?",
                    "question_type": "single_choice",
                    "enum": ["Excellent", "Good", "Fair", "Poor", "Unable"],
                    "description": "Ability to perform physical activities"
                }
            }
        },
        "diabetes_management": {
            "name": "Diabetes Management Assessment",
            "specialty": "Endocrinology",
            "fields": {
                "blood_sugar_control": {
                    "type": "string",
                    "question": "How is the patient's blood sugar control?",
                    "question_type": "single_choice",
                    "enum": ["Excellent", "Good", "Fair", "Poor"],
                    "description": "Overall glucose management"
                },
                "hypoglycemic_episodes": {
                    "type": "boolean",
                    "question": "Has the patient experienced low blood sugar episodes?",
                    "question_type": "yes_no",
                    "description": "Recent hypoglycemic events"
                },
                "medication_adherence": {
                    "type": "string",
                    "question": "How well is the patient following their medication regimen?",
                    "question_type": "single_choice",
                    "enum": ["Always", "Usually", "Sometimes", "Rarely", "Never"],
                    "description": "Compliance with diabetes medications"
                },
                "diet_compliance": {
                    "type": "string",
                    "question": "How well is the patient following their diabetic diet?",
                    "question_type": "single_choice",
                    "enum": ["Excellent", "Good", "Fair", "Poor"],
                    "description": "Adherence to dietary recommendations"
                }
            }
        }
    }


def export_assessment_to_ehr_format(
    assessment_result: Dict[str, Any],
    ehr_format: str = "hl7_fhir"
) -> Dict[str, Any]:
    """
    Export assessment results to standard EHR formats.
    
    Args:
        assessment_result: Assessment results to export
        ehr_format: Target EHR format (hl7_fhir, ccda, etc.)
        
    Returns:
        Assessment data in requested EHR format
    """
    if ehr_format == "hl7_fhir":
        return _convert_to_fhir_observation(assessment_result)
    else:
        raise ValueError(f"Unsupported EHR format: {ehr_format}")


def _convert_to_fhir_observation(assessment_result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert assessment to HL7 FHIR Observation format."""
    metadata = assessment_result.get("assessment_metadata", {})
    responses = assessment_result.get("assessment_responses", {})
    
    fhir_observation = {
        "resourceType": "Observation",
        "id": f"assessment-{metadata.get('patient_id', 'unknown')}",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "survey",
                        "display": "Survey"
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "72133-2",
                    "display": "Clinical assessment"
                }
            ]
        },
        "subject": {
            "reference": f"Patient/{metadata.get('patient_id', 'unknown')}"
        },
        "effectiveDateTime": metadata.get("analysis_timestamp"),
        "performer": [
            {
                "reference": f"Practitioner/{metadata.get('care_staff_id', 'unknown')}"
            }
        ],
        "component": []
    }
    
    # Add assessment responses as components
    for field_name, response in responses.items():
        component = {
            "code": {
                "text": response["question"]
            },
            "valueString": str(response["extracted_value"]) if response["extracted_value"] is not None else "Not answered"
        }
        
        # Add confidence as extension
        if "confidence" in response:
            component["extension"] = [
                {
                    "url": "http://example.com/fhir/StructureDefinition/confidence-score",
                    "valueDecimal": response["confidence"]
                }
            ]
        
        fhir_observation["component"].append(component)
    
    return fhir_observation
