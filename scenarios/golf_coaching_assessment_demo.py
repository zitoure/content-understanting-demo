"""
Golf Coaching Assessment Demo v2 for Azure AI Content Understanding

This system demonstrates how golf coaching conversations can be automatically
analyzed to extract structured assessment data using AI content understanding.

Features:
- Audio transcription and analysis of golf coaching sessions
- Automatic extraction of questions and answers
- Support for skill assessment and feedback analysis
- Confidence scoring for extracted information
- Structured output for coaching review
"""

import os
import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Any
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


class GolfCoachingAnalyzer:
    """
    Analyzes golf coaching conversations to automatically extract structured assessment data.
    Designed for golf coaching sessions with appropriate analysis and feedback extraction.
    """
    
    def __init__(self, client: AzureContentUnderstandingClient):
        self.client = client
        self.output_dir = Path(os.getenv("DEFAULT_OUTPUT_DIR", "./output"))
        self.assessment_output_dir = self.output_dir / "golf_assessments"
        self.assessment_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard assessment templates
        self.assessment_templates = self._load_assessment_templates()
    
    def _load_assessment_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load standard golf coaching assessment templates."""
        templates = {
            "swing_analysis": {
                "name": "Golf Swing Analysis Assessment",
                "version": "1.0",
                "fields": {
                    "swing_technique": {
                        "type": "string",
                        "question": "How would you describe the player's swing technique?",
                        "question_type": "open_ended",
                        "description": "Overall assessment of the player's swing mechanics"
                    },
                    "grip_assessment": {
                        "type": "string",
                        "question": "What is the assessment of the player's grip?",
                        "question_type": "single_choice",
                        "enum": ["Excellent", "Good", "Fair", "Needs Improvement", "Poor"],
                        "description": "Player's grip technique and positioning"
                    },
                    "stance_evaluation": {
                        "type": "string",
                        "question": "How is the player's stance?",
                        "question_type": "single_choice",
                        "enum": ["Excellent", "Good", "Fair", "Needs Improvement", "Poor"],
                        "description": "Player's stance and positioning"
                    },
                    "areas_for_improvement": {
                        "type": "array",
                        "question": "What areas need improvement?",
                        "question_type": "multiple_choice",
                        "items": {"type": "string"},
                        "possible_values": [
                            "Grip technique",
                            "Stance alignment",
                            "Backswing",
                            "Downswing",
                            "Follow-through",
                            "Weight transfer",
                            "Tempo",
                            "Ball position"
                        ],
                        "description": "Specific areas identified for improvement"
                    },
                    "coach_feedback": {
                        "type": "string",
                        "question": "What feedback did the coach provide?",
                        "question_type": "open_ended",
                        "description": "Detailed coaching feedback and recommendations"
                    },
                    "player_questions": {
                        "type": "array",
                        "question": "What questions did the player ask?",
                        "question_type": "open_ended",
                        "items": {"type": "string"},
                        "description": "Questions asked by the player during the session"
                    },
                    "practice_recommendations": {
                        "type": "string",
                        "question": "What practice recommendations were given?",
                        "question_type": "open_ended",
                        "description": "Specific practice drills and exercises recommended"
                    },
                    "session_rating": {
                        "type": "string",
                        "question": "How would you rate this coaching session?",
                        "question_type": "single_choice",
                        "enum": ["Excellent", "Good", "Fair", "Needs Improvement"],
                        "description": "Overall session effectiveness rating"
                    },
                    "follow_up_needed": {
                        "type": "boolean",
                        "question": "Is follow-up coaching needed?",
                        "question_type": "yes_no",
                        "description": "Whether additional coaching sessions are recommended"
                    },
                    "additional_notes": {
                        "type": "string",
                        "question": "Additional coaching observations or notes",
                        "question_type": "open_ended",
                        "description": "Any additional relevant information from the session"
                    }
                }
            },
            
            "performance_review": {
                "name": "Golf Performance Review",
                "version": "1.0",
                "fields": {
                    "performance_summary": {
                        "type": "string",
                        "question": "What is the overall performance summary?",
                        "question_type": "open_ended",
                        "description": "Summary of the player's performance during the session"
                    },
                    "skill_level": {
                        "type": "string",
                        "question": "What is the player's current skill level?",
                        "question_type": "single_choice",
                        "enum": ["Beginner", "Intermediate", "Advanced", "Expert"],
                        "description": "Current skill level assessment"
                    },
                    "strengths_identified": {
                        "type": "array",
                        "question": "What strengths were identified?",
                        "question_type": "multiple_choice",
                        "items": {"type": "string"},
                        "possible_values": [
                            "Consistent swing",
                            "Good tempo",
                            "Proper alignment",
                            "Mental focus",
                            "Course management",
                            "Short game",
                            "Putting",
                            "Physical fitness"
                        ],
                        "description": "Player's identified strengths"
                    },
                    "goals_discussed": {
                        "type": "string",
                        "question": "What goals were discussed?",
                        "question_type": "open_ended",
                        "description": "Goals and objectives discussed during the session"
                    }
                }
            },
            
            "full_analysis": {
                "name": "Comprehensive Golf Coaching Analysis",
                "version": "1.0",
                "fields": {
                    # Swing Analysis Fields
                    "swing_technique": {
                        "type": "string",
                        "question": "How would you describe the player's swing technique?",
                        "question_type": "open_ended",
                        "description": "Overall assessment of the player's swing mechanics"
                    },
                    "grip_assessment": {
                        "type": "string",
                        "question": "What is the assessment of the player's grip?",
                        "question_type": "single_choice",
                        "enum": ["Excellent", "Good", "Fair", "Needs Improvement", "Poor"],
                        "description": "Player's grip technique and positioning"
                    },
                    "stance_evaluation": {
                        "type": "string",
                        "question": "How is the player's stance?",
                        "question_type": "single_choice",
                        "enum": ["Excellent", "Good", "Fair", "Needs Improvement", "Poor"],
                        "description": "Player's stance and positioning"
                    },
                    "areas_for_improvement": {
                        "type": "array",
                        "question": "What areas need improvement?",
                        "question_type": "multiple_choice",
                        "items": {"type": "string"},
                        "possible_values": [
                            "Grip technique",
                            "Stance alignment",
                            "Backswing",
                            "Downswing",
                            "Follow-through",
                            "Weight transfer",
                            "Tempo",
                            "Ball position",
                            "Chipping",
                            "Putting",
                            "Mental game",
                            "Course management"
                        ],
                        "description": "Specific areas identified for improvement"
                    },
                    "coach_feedback": {
                        "type": "string",
                        "question": "What feedback did the coach provide?",
                        "question_type": "open_ended",
                        "description": "Detailed coaching feedback and recommendations"
                    },
                    "practice_recommendations": {
                        "type": "string",
                        "question": "What practice recommendations were given?",
                        "question_type": "open_ended",
                        "description": "Specific practice drills and exercises recommended"
                    },
                    # Performance Review Fields
                    "performance_summary": {
                        "type": "string",
                        "question": "What is the overall performance summary?",
                        "question_type": "open_ended",
                        "description": "Summary of the player's performance during the session"
                    },
                    "skill_level": {
                        "type": "string",
                        "question": "What is the player's current skill level?",
                        "question_type": "single_choice",
                        "enum": ["Beginner", "Intermediate", "Advanced", "Expert"],
                        "description": "Current skill level assessment"
                    },
                    "strengths_identified": {
                        "type": "array",
                        "question": "What strengths were identified?",
                        "question_type": "multiple_choice",
                        "items": {"type": "string"},
                        "possible_values": [
                            "Consistent swing",
                            "Good tempo",
                            "Proper alignment",
                            "Mental focus",
                            "Course management",
                            "Short game",
                            "Putting",
                            "Physical fitness",
                            "Good distance with driver",
                            "Dedication to practice"
                        ],
                        "description": "Player's identified strengths"
                    },
                    "goals_discussed": {
                        "type": "string",
                        "question": "What goals were discussed?",
                        "question_type": "open_ended",
                        "description": "Goals and objectives discussed during the session"
                    },
                    # Additional Comprehensive Fields
                    "current_handicap": {
                        "type": "string",
                        "question": "What is the player's current handicap?",
                        "question_type": "open_ended",
                        "description": "Player's current golf handicap"
                    },
                    "years_experience": {
                        "type": "integer",
                        "question": "How many years has the player been playing golf?",
                        "question_type": "open_ended",
                        "description": "Number of years the player has been playing golf"
                    },
                    "strongest_club": {
                        "type": "string",
                        "question": "What is the player's strongest club?",
                        "question_type": "open_ended",
                        "description": "Club the player feels most confident using"
                    },
                    "weakest_area": {
                        "type": "string",
                        "question": "What is the player's weakest area of their game?",
                        "question_type": "open_ended",
                        "description": "Area of the game that needs the most improvement"
                    },
                    "mental_game_assessment": {
                        "type": "string",
                        "question": "How does the player handle pressure and mental aspects?",
                        "question_type": "open_ended",
                        "description": "Assessment of player's mental game and pressure handling"
                    },
                    "fitness_routine": {
                        "type": "string",
                        "question": "Does the player have a fitness routine?",
                        "question_type": "open_ended",
                        "description": "Player's current fitness routine for golf"
                    },
                    "practice_schedule": {
                        "type": "string",
                        "question": "What is the player's practice schedule?",
                        "question_type": "open_ended",
                        "description": "How often and how long the player practices"
                    },
                    "competitive_goals": {
                        "type": "string",
                        "question": "What are the player's competitive goals?",
                        "question_type": "open_ended",
                        "description": "Player's goals for competition and improvement"
                    },
                    "session_rating": {
                        "type": "string",
                        "question": "How would you rate this coaching session?",
                        "question_type": "single_choice",
                        "enum": ["Excellent", "Good", "Fair", "Needs Improvement"],
                        "description": "Overall session effectiveness rating"
                    },
                    "follow_up_needed": {
                        "type": "boolean",
                        "question": "Is follow-up coaching needed?",
                        "question_type": "yes_no",
                        "description": "Whether additional coaching sessions are recommended"
                    },
                    "additional_notes": {
                        "type": "string",
                        "question": "Additional coaching observations or notes",
                        "question_type": "open_ended",
                        "description": "Any additional relevant information from the session"
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
        """Create or get an analyzer for the specified assessment type."""
        try:
            if assessment_type not in self.assessment_templates:
                raise ValueError(f"Unknown assessment type: {assessment_type}")
            
            template = self.assessment_templates[assessment_type]
            
            if analyzer_id is None:
                analyzer_id = f"golf_{assessment_type}_analyzer"
            
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
        
        # Create analyzer configuration
        analyzer_config = {
            "description": f"Analyzes golf coaching conversations for {template['name']}",
            "baseAnalyzerId": "prebuilt-callCenter",
            "config": {
                "locales": ["en-US"],
                "returnDetails": True,
                "disableContentFiltering": False
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
        
        logger.info(f"Analyzer created successfully: {result}")
        return analyzer_id
    
    def analyze_golf_conversation(
        self,
        audio_url: str,
        assessment_type: str = "swing_analysis",
        player_id: Optional[str] = None,
        coach_id: Optional[str] = None,
        session_metadata: Optional[Dict[str, Any]] = None,
        analyzer_id: Optional[str] = None,
        force_overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze a golf coaching conversation and extract assessment data.
        
        Args:
            audio_url: URL of the conversation audio
            assessment_type: Type of assessment to perform
            player_id: Optional player identifier
            coach_id: Optional coach identifier  
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
        
        try:
            # Analyze the conversation
            logger.info(f"Starting analysis of golf conversation for {assessment_type}")
            analysis_request = self.client.analyze_content(
                analyzer_id=analyzer_id,
                content_url=audio_url
            )
            
            logger.info(f"Analysis request response: {analysis_request}")
            
            # Get request ID
            request_id = analysis_request.get("request_id")
            if not request_id:
                raise ValueError(f"No request ID found in response: {analysis_request}")
            
            logger.info(f"Analysis started with request ID: {request_id}")
            
            # Wait for completion
            result = self.client.wait_for_analysis_completion(
                request_id,
                max_wait_time=300,  # 5 minutes
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
            player_id,
            coach_id,
            session_metadata
        )
        
        return assessment_result
    
    def _process_assessment_results(
        self,
        raw_result: Dict[str, Any],
        assessment_type: str,
        player_id: Optional[str],
        coach_id: Optional[str],
        session_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process raw analysis results into structured assessment format."""
        template = self.assessment_templates[assessment_type]
        
        # Extract conversation transcript
        transcript_text = extract_transcript_text(raw_result)
        
        # Extract structured data from analysis result
        extracted_data = raw_result.get("result", {}).get("contents", [])
        
        # Initialize assessment result
        assessment_result = {
            "assessment_info": {
                "assessment_type": assessment_type,
                "template_name": template["name"],
                "template_version": template["version"],
                "analysis_timestamp": datetime.now().isoformat(),
                "player_id": player_id,
                "coach_id": coach_id,
                "session_metadata": session_metadata or {}
            },
            "conversation": {
                "transcript": transcript_text,
                "audio_url": raw_result.get("audio_url", ""),
                "duration_seconds": raw_result.get("duration_seconds")
            },
            "extracted_fields": {},
            "raw_analysis": raw_result
        }
        
        # Process each field from the template
        for field_name, field_config in template["fields"].items():
            extracted_value = self._extract_field_value(
                raw_result, field_name, field_config
            )
            
            assessment_result["extracted_fields"][field_name] = extracted_value
        
        # Save assessment result
        self._save_assessment_result(assessment_result, assessment_type)
        
        return assessment_result
    
    def _extract_field_value(
        self,
        raw_result: Dict[str, Any],
        field_name: str,
        field_config: Dict[str, Any]
    ) -> Any:
        """Extract a specific field value from the analysis results."""
        # Navigate to the fields in the analysis result
        try:
            contents = raw_result.get("result", {}).get("contents", [])
            if not contents:
                logger.warning("No contents found in analysis result")
                return self._get_default_value(field_config["type"])
            
            # Get the first content item (should contain the fields)
            content_item = contents[0]
            fields = content_item.get("fields", {})
            
            if field_name not in fields:
                logger.warning(f"Field '{field_name}' not found in analysis results")
                return self._get_default_value(field_config["type"])
            
            field_data = fields[field_name]
            
            # Extract value based on field type in the response
            if field_data.get("type") == "string":
                value = field_data.get("valueString", "")
            elif field_data.get("type") == "array":
                value_array = field_data.get("valueArray", [])
                value = [item.get("valueString", "") for item in value_array]
            elif field_data.get("type") == "boolean":
                value = field_data.get("valueBoolean")
            elif field_data.get("type") == "integer":
                value = field_data.get("valueInteger")
            elif field_data.get("type") == "number":
                value = field_data.get("valueNumber")
            else:
                # Fallback for other types
                value = field_data.get("valueString", "")
            
            # Validate and format based on expected field type
            if field_config["type"] == "array" and not isinstance(value, list):
                value = [str(value)] if value else []
            elif field_config["type"] == "boolean":
                if isinstance(value, str):
                    value = value.lower() in ["true", "yes", "1"]
            elif field_config["type"] == "integer":
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    value = 0
            
            return value
            
        except Exception as e:
            logger.error(f"Error extracting field '{field_name}': {e}")
            return self._get_default_value(field_config["type"])
    
    def _get_default_value(self, field_type: str) -> Any:
        """Get default value for a field type."""
        defaults = {
            "string": "",
            "array": [],
            "boolean": False,
            "integer": 0
        }
        return defaults.get(field_type, None)
    
    def _save_assessment_result(self, assessment_result: Dict[str, Any], assessment_type: str):
        """Save assessment result to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"golf_{assessment_type}_{timestamp}.json"
        filepath = self.assessment_output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(assessment_result, f, indent=2, default=str)
        
        logger.info(f"Assessment result saved to: {filepath}")
    
    def run_demo_scenarios(self):
        """Run demo scenarios with sample data."""
        logger.info("Starting Golf Coaching Assessment Demo v2")
        
        # Demo scenarios
        scenarios = [
            {
                "name": "Comprehensive Golf Analysis",
                "audio_url": "https://github.com/zitoure/content-understanting-demo/raw/main/audio/golf_coaching_en-US.wav",
                "assessment_type": "full_analysis",
                "player_id": "player_001",
                "coach_id": "coach_001"
            },
            {
                "name": "Beginner Swing Analysis",
                "audio_url": "https://github.com/zitoure/content-understanting-demo/raw/main/audio/golf_coaching_en-US.wav",
                "assessment_type": "swing_analysis",
                "player_id": "player_002",
                "coach_id": "coach_001"
            },
            {
                "name": "Advanced Performance Review",
                "audio_url": "https://github.com/zitoure/content-understanting-demo/raw/main/audio/golf_coaching_en-US.wav",
                "assessment_type": "performance_review", 
                "player_id": "player_003",
                "coach_id": "coach_001"
            }
        ]
        
        for scenario in scenarios:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running scenario: {scenario['name']}")
            logger.info(f"{'='*50}")
            
            try:
                result = self.analyze_golf_conversation(
                    audio_url=scenario["audio_url"],
                    assessment_type=scenario["assessment_type"],
                    player_id=scenario["player_id"],
                    coach_id=scenario["coach_id"],
                    session_metadata={
                        "scenario": scenario["name"],
                        "demo_mode": True
                    }
                )
                
                # Display results summary
                self._display_assessment_summary(result)
                
            except Exception as e:
                logger.error(f"Error in scenario '{scenario['name']}': {e}")
                continue
    
    def _display_assessment_summary(self, result: Dict[str, Any]):
        """Display a summary of the assessment results."""
        print(f"\n📊 Assessment Summary:")
        print(f"Assessment Type: {result['assessment_info']['template_name']}")
        print(f"Analysis Time: {result['assessment_info']['analysis_timestamp']}")
        
        print(f"\n📝 Extracted Fields:")
        for field_name, value in result['extracted_fields'].items():
            print(f"  • {field_name}: {value}")
        
        print(f"\n📄 Transcript Preview:")
        transcript = result['conversation']['transcript']
        preview = transcript[:200] + "..." if len(transcript) > 200 else transcript
        print(f"  {preview}")


def main():
    """Main function to run the golf coaching assessment demo."""
    parser = argparse.ArgumentParser(description="Golf Coaching Assessment Demo v2")
    parser.add_argument("--audio-url", type=str, help="URL of audio file to analyze")
    parser.add_argument("--assessment-type", type=str, default="full_analysis",
                       choices=["swing_analysis", "performance_review", "full_analysis"],
                       help="Type of assessment to perform")
    parser.add_argument("--player-id", type=str, help="Player identifier")
    parser.add_argument("--coach-id", type=str, help="Coach identifier")
    parser.add_argument("--demo", action="store_true", help="Run demo scenarios")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Create client
    try:
        client = create_client_from_env()
        analyzer = GolfCoachingAnalyzer(client)
        
        if args.demo:
            # Run demo scenarios
            analyzer.run_demo_scenarios()
        elif args.audio_url:
            # Analyze specific audio file
            logger.info(f"Analyzing audio: {args.audio_url}")
            result = analyzer.analyze_golf_conversation(
                audio_url=args.audio_url,
                assessment_type=args.assessment_type,
                player_id=args.player_id,
                coach_id=args.coach_id
            )
            
            analyzer._display_assessment_summary(result)
            
        else:
            print("Please provide --audio-url or use --demo flag")
            return 1
            
        return 0
        
    except Exception as e:
        logger.error(f"Failed to run golf coaching assessment: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
