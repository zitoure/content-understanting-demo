"""
Golf Coaching Assessment Demo

This demo shows how to use Azure AI Content Understanding to automatically process
streaming audio conversations between golf coaches and athletes, detecting questions
and answers to fill out comprehensive golf performance assessment forms.

The system supports real-time audio streaming and provides structured assessment
data for golf coaching and performance analysis.
"""

import json
import asyncio
import threading
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from content_understanding_client import AzureContentUnderstandingClient
from utils import setup_logging, create_output_directory


class AssessmentSection(Enum):
    """Golf assessment sections."""
    PLAYER_BACKGROUND = "player_background"
    TECHNICAL_SKILLS = "technical_skills"
    MENTAL_GAME = "mental_game"
    PHYSICAL_FITNESS = "physical_fitness"
    COURSE_MANAGEMENT = "course_management"
    GOALS_MOTIVATION = "goals_motivation"


@dataclass
class QuestionAnswerPair:
    """Represents a detected question-answer pair from conversation."""
    question: str
    answer: str
    speaker_roles: Dict[str, str]  # speaker_id -> role (coach/athlete)
    timestamp: float
    confidence: float
    section: Optional[AssessmentSection] = None
    field_mapping: Optional[str] = None


class StreamingAudioProcessor:
    """Handles streaming audio processing and buffering."""
    
    def __init__(self, buffer_duration: float = 30.0, overlap_duration: float = 5.0):
        self.buffer_duration = buffer_duration  # seconds
        self.overlap_duration = overlap_duration
        self.audio_buffers = []
        self.current_buffer = bytearray()
        self.last_process_time = 0
        self.is_streaming = False
        self.processing_callback: Optional[Callable] = None
    
    def start_streaming(self, processing_callback: Callable):
        """Start processing streaming audio."""
        self.processing_callback = processing_callback
        self.is_streaming = True
        self.last_process_time = time.time()
        
        # Start background processing thread
        processing_thread = threading.Thread(target=self._process_audio_loop)
        processing_thread.daemon = True
        processing_thread.start()
    
    def add_audio_chunk(self, audio_chunk: bytes):
        """Add new audio chunk to the buffer."""
        if not self.is_streaming:
            return
        
        self.current_buffer.extend(audio_chunk)
        current_time = time.time()
        
        # Check if it's time to process the buffer
        if current_time - self.last_process_time >= self.buffer_duration:
            self._trigger_processing()
    
    def _trigger_processing(self):
        """Trigger processing of current audio buffer."""
        if len(self.current_buffer) > 0:
            # Create audio buffer for processing
            buffer_copy = bytes(self.current_buffer)
            self.audio_buffers.append({
                'data': buffer_copy,
                'timestamp': time.time(),
                'processed': False
            })
            
            # Keep overlap for context
            overlap_size = int(len(self.current_buffer) * (self.overlap_duration / self.buffer_duration))
            self.current_buffer = self.current_buffer[-overlap_size:] if overlap_size > 0 else bytearray()
            
            self.last_process_time = time.time()
    
    def _process_audio_loop(self):
        """Background loop for processing audio buffers."""
        while self.is_streaming:
            # Process pending buffers
            for buffer_info in self.audio_buffers:
                if not buffer_info['processed'] and self.processing_callback:
                    try:
                        self.processing_callback(buffer_info['data'], buffer_info['timestamp'])
                        buffer_info['processed'] = True
                    except Exception as e:
                        print(f"Error processing audio buffer: {e}")
            
            # Clean up old processed buffers
            self.audio_buffers = [b for b in self.audio_buffers if not b['processed'] or 
                                time.time() - b['timestamp'] < 300]  # Keep for 5 minutes
            
            time.sleep(1)  # Process every second
    
    def stop_streaming(self):
        """Stop streaming audio processing."""
        self.is_streaming = False
        # Process final buffer
        if len(self.current_buffer) > 0:
            self._trigger_processing()


class GolfAssessmentAnalyzer:
    """
    Main class for analyzing golf coaching conversations and creating assessment forms.
    Processes streaming audio to detect questions and answers automatically.
    """
    
    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        """Initialize the golf assessment analyzer."""
        self.client = AzureContentUnderstandingClient(api_key, endpoint)
        self.logger = setup_logging("golf_assessment")
        self.output_dir = create_output_directory("golf_assessments")
        
        # Assessment form templates
        self.assessment_templates = self._create_golf_assessment_templates()
        
        # Question-answer detection
        self.detected_qa_pairs: List[QuestionAnswerPair] = []
        self.conversation_context = {}
        
        # Streaming components
        self.audio_processor = StreamingAudioProcessor()
        self.current_session_id = None
        
        # Speaker identification
        self.speaker_roles = {}  # speaker_id -> role mapping
        
        self.logger.info("Golf Assessment Analyzer initialized")
    
    def _create_golf_assessment_templates(self) -> Dict[str, Any]:
        """Create comprehensive golf assessment form templates."""
        return {
            "comprehensive_golf_assessment": {
                "name": "Comprehensive Golf Performance Assessment",
                "description": "Complete evaluation of golfer's skills, mental game, and development areas",
                "sections": {
                    "player_background": {
                        "title": "Player Background & Experience",
                        "fields": {
                            "years_playing": {
                                "type": "integer",
                                "question": "How long have you been playing golf?",
                                "question_type": "open_ended",
                                "description": "Total years of golf experience",
                                "keywords": ["years", "playing", "started", "began", "experience"]
                            },
                            "current_handicap": {
                                "type": "string",
                                "question": "What's your current handicap?",
                                "question_type": "open_ended", 
                                "description": "Official handicap index",
                                "keywords": ["handicap", "index", "USGA", "scoring"]
                            },
                            "rounds_per_month": {
                                "type": "integer",
                                "question": "How many rounds do you typically play per month?",
                                "question_type": "open_ended",
                                "description": "Frequency of play",
                                "keywords": ["rounds", "play", "month", "frequency", "often"]
                            },
                            "home_course": {
                                "type": "string",
                                "question": "Where do you usually play? What's your home course?",
                                "question_type": "open_ended",
                                "description": "Primary golf course or club",
                                "keywords": ["home course", "club", "usually play", "member"]
                            },
                            "previous_instruction": {
                                "type": "boolean",
                                "question": "Have you had professional golf instruction before?",
                                "question_type": "yes_no",
                                "description": "Prior coaching experience",
                                "keywords": ["instruction", "lessons", "coach", "professional", "taught"]
                            }
                        }
                    },
                    "technical_skills": {
                        "title": "Technical Skills Assessment",
                        "fields": {
                            "strongest_club": {
                                "type": "string",
                                "question": "What club do you feel most confident with?",
                                "question_type": "open_ended",
                                "description": "Most reliable club in bag",
                                "keywords": ["confident", "best", "strongest", "favorite", "reliable"]
                            },
                            "weakest_area": {
                                "type": "string",
                                "question": "What part of your game needs the most work?",
                                "question_type": "open_ended",
                                "description": "Primary area for improvement",
                                "keywords": ["struggle", "difficult", "worst", "needs work", "improve"]
                            },
                            "driving_accuracy": {
                                "type": "string",
                                "question": "How would you rate your driving accuracy?",
                                "question_type": "single_choice",
                                "enum": ["Excellent", "Good", "Average", "Below Average", "Poor"],
                                "description": "Self-assessment of driving accuracy",
                                "keywords": ["driving", "accuracy", "fairways", "straight"]
                            },
                            "short_game_confidence": {
                                "type": "string",
                                "question": "How confident are you with your short game?",
                                "question_type": "single_choice",
                                "enum": ["Very Confident", "Confident", "Somewhat Confident", "Not Confident"],
                                "description": "Short game self-assessment",
                                "keywords": ["short game", "chipping", "pitching", "wedges", "around green"]
                            },
                            "putting_consistency": {
                                "type": "string",
                                "question": "How consistent is your putting?",
                                "question_type": "single_choice",
                                "enum": ["Very Consistent", "Consistent", "Inconsistent", "Very Inconsistent"],
                                "description": "Putting performance consistency",
                                "keywords": ["putting", "putts", "green", "consistency"]
                            }
                        }
                    },
                    "mental_game": {
                        "title": "Mental Game & Course Management",
                        "fields": {
                            "pre_shot_routine": {
                                "type": "boolean",
                                "question": "Do you have a consistent pre-shot routine?",
                                "question_type": "yes_no",
                                "description": "Pre-shot routine consistency",
                                "keywords": ["pre-shot", "routine", "process", "preparation"]
                            },
                            "pressure_handling": {
                                "type": "string",
                                "question": "How do you handle pressure situations on the course?",
                                "question_type": "open_ended",
                                "description": "Pressure management approach",
                                "keywords": ["pressure", "nervous", "clutch", "important", "stress"]
                            },
                            "course_strategy": {
                                "type": "string",
                                "question": "Do you typically play aggressive or conservative golf?",
                                "question_type": "single_choice",
                                "enum": ["Very Aggressive", "Aggressive", "Balanced", "Conservative", "Very Conservative"],
                                "description": "Overall playing strategy",
                                "keywords": ["aggressive", "conservative", "strategy", "risk", "safe"]
                            },
                            "mental_toughness": {
                                "type": "string",
                                "question": "How do you bounce back from bad shots or holes?",
                                "question_type": "open_ended",
                                "description": "Resilience and mental recovery",
                                "keywords": ["bounce back", "bad shots", "recovery", "focus", "reset"]
                            }
                        }
                    },
                    "physical_fitness": {
                        "title": "Physical Fitness & Conditioning",
                        "fields": {
                            "flexibility": {
                                "type": "string",
                                "question": "How would you rate your flexibility?",
                                "question_type": "single_choice",
                                "enum": ["Excellent", "Good", "Average", "Below Average", "Poor"],
                                "description": "Overall flexibility assessment",
                                "keywords": ["flexibility", "flexible", "stretch", "range of motion"]
                            },
                            "fitness_routine": {
                                "type": "boolean",
                                "question": "Do you have a regular fitness routine?",
                                "question_type": "yes_no",
                                "description": "Regular exercise habits",
                                "keywords": ["fitness", "exercise", "workout", "gym", "training"]
                            },
                            "injuries_limitations": {
                                "type": "string",
                                "question": "Do you have any injuries or physical limitations?",
                                "question_type": "open_ended",
                                "description": "Physical constraints or concerns",
                                "keywords": ["injury", "pain", "limitation", "problem", "hurt"]
                            },
                            "stamina_endurance": {
                                "type": "string",
                                "question": "How's your energy level during a full round?",
                                "question_type": "single_choice",
                                "enum": ["Strong Throughout", "Good", "Fades Late", "Gets Tired", "Poor Endurance"],
                                "description": "Endurance during play",
                                "keywords": ["energy", "tired", "endurance", "stamina", "18 holes"]
                            }
                        }
                    },
                    "goals_motivation": {
                        "title": "Goals & Motivation",
                        "fields": {
                            "primary_goal": {
                                "type": "string",
                                "question": "What's your main goal for improving your golf game?",
                                "question_type": "open_ended",
                                "description": "Primary improvement objective",
                                "keywords": ["goal", "want", "improve", "better", "achieve"]
                            },
                            "target_handicap": {
                                "type": "string",
                                "question": "What handicap would you like to achieve?",
                                "question_type": "open_ended",
                                "description": "Target handicap goal",
                                "keywords": ["target", "goal", "handicap", "achieve", "reach"]
                            },
                            "motivation_level": {
                                "type": "string",
                                "question": "How motivated are you to practice and improve?",
                                "question_type": "single_choice",
                                "enum": ["Extremely Motivated", "Very Motivated", "Motivated", "Somewhat Motivated", "Not Very Motivated"],
                                "description": "Level of commitment to improvement",
                                "keywords": ["motivated", "practice", "committed", "dedicated", "willing"]
                            },
                            "practice_time": {
                                "type": "string",
                                "question": "How much time can you dedicate to practice per week?",
                                "question_type": "open_ended",
                                "description": "Available practice time commitment",
                                "keywords": ["practice", "time", "hours", "week", "available"]
                            },
                            "competitive_interest": {
                                "type": "boolean",
                                "question": "Are you interested in competitive golf or tournaments?",
                                "question_type": "yes_no",
                                "description": "Interest in competitive play",
                                "keywords": ["competitive", "tournament", "compete", "events"]
                            }
                        }
                    }
                }
            }
        }
    
    async def start_streaming_assessment(
        self,
        athlete_id: str,
        coach_id: str,
        assessment_type: str = "comprehensive_golf_assessment",
        session_notes: Optional[str] = None
    ) -> str:
        """
        Start a streaming assessment session.
        
        Args:
            athlete_id: Unique identifier for the athlete
            coach_id: Unique identifier for the coach
            assessment_type: Type of assessment to conduct
            session_notes: Optional notes about the session
            
        Returns:
            Session ID for tracking
        """
        session_id = f"golf_session_{int(time.time())}_{athlete_id}"
        self.current_session_id = session_id
        
        # Initialize session data
        self.conversation_context = {
            "session_id": session_id,
            "athlete_id": athlete_id,
            "coach_id": coach_id,
            "assessment_type": assessment_type,
            "start_time": datetime.now().isoformat(),
            "session_notes": session_notes,
            "qa_pairs": [],
            "current_responses": {}
        }
        
        # Set up speaker roles
        self.speaker_roles = {
            "speaker_1": "coach",
            "speaker_2": "athlete"
        }
        
        # Start streaming processor
        self.audio_processor.start_streaming(self._process_audio_chunk)
        
        self.logger.info(f"Started streaming assessment session {session_id}")
        return session_id
    
    def add_audio_stream_chunk(self, audio_chunk: bytes):
        """Add audio chunk to streaming processor."""
        if self.current_session_id:
            self.audio_processor.add_audio_chunk(audio_chunk)
    
    async def _process_audio_chunk(self, audio_data: bytes, timestamp: float):
        """Process audio chunk and extract question-answer pairs."""
        try:
            self.logger.info(f"Processing audio chunk from timestamp {timestamp}")
            
            # Create temporary audio file for processing
            temp_audio_file = self.output_dir / f"temp_audio_{int(timestamp)}.wav"
            
            # Save audio chunk (real audio data from streaming)
            with open(temp_audio_file, 'wb') as f:
                f.write(audio_data)
            
            # Analyze audio with Content Understanding
            result = await self._analyze_audio_content(str(temp_audio_file))
            
            # Extract question-answer pairs
            qa_pairs = self._extract_qa_pairs_from_audio(result, timestamp)
            
            # Process each Q&A pair
            for qa_pair in qa_pairs:
                await self._process_qa_pair(qa_pair)
            
            # Clean up temp file
            if temp_audio_file.exists():
                temp_audio_file.unlink()
                
        except Exception as e:
            self.logger.error(f"Error processing audio chunk: {e}")
    
    async def _analyze_audio_content(self, audio_file_path: str) -> Dict[str, Any]:
        """Analyze audio content using Azure AI Content Understanding."""
        # Create custom analyzer for golf coaching conversations
        analyzer_name = f"golf_conversation_analyzer_{int(time.time())}"
        
        conversation_schema = {
            "fields": {
                "conversation_transcript": {
                    "type": "string",
                    "description": "Full conversation transcript with speaker identification"
                },
                "speaker_segments": {
                    "type": "array",
                    "description": "Individual speaker segments with timestamps",
                    "items": {
                        "type": "object",
                        "properties": {
                            "speaker_id": {"type": "string"},
                            "text": {"type": "string"},
                            "start_time": {"type": "number"},
                            "end_time": {"type": "number"}
                        }
                    }
                },
                "questions_detected": {
                    "type": "array",
                    "description": "Detected questions in the conversation",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_text": {"type": "string"},
                            "speaker_id": {"type": "string"},
                            "timestamp": {"type": "number"}
                        }
                    }
                },
                "responses_detected": {
                    "type": "array", 
                    "description": "Detected responses to questions",
                    "items": {
                        "type": "object",
                        "properties": {
                            "response_text": {"type": "string"},
                            "speaker_id": {"type": "string"},
                            "timestamp": {"type": "number"},
                            "related_question": {"type": "string"}
                        }
                    }
                }
            }
        }
        
        # Create analyzer
        analyzer_result = self.client.create_analyzer(
            analyzer_name=analyzer_name,
            field_schema=conversation_schema,
            analyzer_description="Analyzes golf coaching conversations to extract questions and answers"
        )
        
        if not analyzer_result["success"]:
            raise Exception(f"Failed to create analyzer: {analyzer_result['error']}")
        
        # Analyze audio
        analysis_result = self.client.analyze_content(
            file_path=audio_file_path,
            analyzer_name=analyzer_name,
            content_type="audio"
        )
        
        return analysis_result
    
    def _extract_qa_pairs_from_audio(self, analysis_result: Dict[str, Any], timestamp: float) -> List[QuestionAnswerPair]:
        """Extract question-answer pairs from audio analysis results."""
        qa_pairs = []
        
        if not analysis_result.get("success"):
            return qa_pairs
        
        extracted_data = analysis_result.get("extracted_data", {})
        questions = extracted_data.get("questions_detected", [])
        responses = extracted_data.get("responses_detected", [])
        
        # Match questions with responses
        for question_data in questions:
            question_text = question_data.get("question_text", "")
            question_speaker = question_data.get("speaker_id", "")
            question_time = question_data.get("timestamp", timestamp)
            
            # Find matching response
            matching_response = None
            for response_data in responses:
                if (response_data.get("related_question") == question_text or
                    abs(response_data.get("timestamp", 0) - question_time) < 30):  # Within 30 seconds
                    matching_response = response_data
                    break
            
            if matching_response:
                qa_pair = QuestionAnswerPair(
                    question=question_text,
                    answer=matching_response.get("response_text", ""),
                    speaker_roles={
                        question_speaker: self.speaker_roles.get(question_speaker, "unknown"),
                        matching_response.get("speaker_id", ""): self.speaker_roles.get(
                            matching_response.get("speaker_id", ""), "unknown"
                        )
                    },
                    timestamp=question_time,
                    confidence=0.8  # Default confidence, could be improved with actual confidence scores
                )
                
                qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    async def _process_qa_pair(self, qa_pair: QuestionAnswerPair):
        """Process a detected question-answer pair and map to assessment fields."""
        try:
            # Add to detected pairs
            self.detected_qa_pairs.append(qa_pair)
            
            # Map to assessment fields
            field_mapping = self._map_qa_to_assessment_field(qa_pair)
            
            if field_mapping:
                qa_pair.field_mapping = field_mapping["field_name"]
                qa_pair.section = AssessmentSection(field_mapping["section"])
                
                # Update assessment responses
                self._update_assessment_response(field_mapping, qa_pair)
                
                self.logger.info(f"Mapped Q&A to field: {field_mapping['field_name']}")
            
            # Save to conversation context
            self.conversation_context["qa_pairs"].append({
                "question": qa_pair.question,
                "answer": qa_pair.answer,
                "timestamp": qa_pair.timestamp,
                "field_mapping": qa_pair.field_mapping,
                "section": qa_pair.section.value if qa_pair.section else None
            })
            
        except Exception as e:
            self.logger.error(f"Error processing Q&A pair: {e}")
    
    def _map_qa_to_assessment_field(self, qa_pair: QuestionAnswerPair) -> Optional[Dict[str, Any]]:
        """Map question-answer pair to specific assessment field."""
        assessment_template = self.assessment_templates.get(
            self.conversation_context["assessment_type"], {}
        )
        
        sections = assessment_template.get("sections", {})
        
        # Search through all fields to find best match
        best_match = None
        best_score = 0
        
        for section_name, section_data in sections.items():
            for field_name, field_config in section_data.get("fields", {}).items():
                # Check keywords match
                keywords = field_config.get("keywords", [])
                question_lower = qa_pair.question.lower()
                
                score = 0
                for keyword in keywords:
                    if keyword.lower() in question_lower:
                        score += 1
                
                # Check question similarity (basic approach)
                template_question = field_config.get("question", "").lower()
                if template_question in question_lower or question_lower in template_question:
                    score += 3
                
                # Normalize score
                if len(keywords) > 0:
                    score = score / len(keywords)
                
                if score > best_score and score > 0.3:  # Minimum threshold
                    best_score = score
                    best_match = {
                        "section": section_name,
                        "field_name": field_name,
                        "field_config": field_config,
                        "match_score": score
                    }
        
        return best_match
    
    def _update_assessment_response(self, field_mapping: Dict[str, Any], qa_pair: QuestionAnswerPair):
        """Update assessment response with extracted answer."""
        field_name = field_mapping["field_name"]
        field_config = field_mapping["field_config"]
        
        # Process answer based on field type
        processed_answer = self._process_answer_value(qa_pair.answer, field_config)
        
        # Update current responses
        self.conversation_context["current_responses"][field_name] = {
            "question": field_config["question"],
            "extracted_value": processed_answer,
            "original_question": qa_pair.question,
            "original_answer": qa_pair.answer,
            "confidence": qa_pair.confidence,
            "timestamp": qa_pair.timestamp,
            "question_type": field_config["question_type"],
            "section": field_mapping["section"]
        }
    
    def _process_answer_value(self, answer: str, field_config: Dict[str, Any]) -> Any:
        """Process answer value according to field type and constraints."""
        field_type = field_config.get("type", "string")
        question_type = field_config.get("question_type", "open_ended")
        
        if question_type == "yes_no":
            # Extract boolean from answer
            answer_lower = answer.lower()
            yes_indicators = ["yes", "yeah", "yep", "sure", "definitely", "absolutely", "correct"]
            no_indicators = ["no", "nope", "not", "never", "negative"]
            
            for indicator in yes_indicators:
                if indicator in answer_lower:
                    return True
            
            for indicator in no_indicators:
                if indicator in answer_lower:
                    return False
            
            return None  # Unclear answer
        
        elif question_type == "single_choice" and "enum" in field_config:
            # Match answer to enum values
            enum_values = field_config["enum"]
            answer_lower = answer.lower()
            
            for enum_value in enum_values:
                if enum_value.lower() in answer_lower:
                    return enum_value
            
            # Try partial matching
            for enum_value in enum_values:
                enum_words = enum_value.lower().split()
                if any(word in answer_lower for word in enum_words):
                    return enum_value
            
            return answer  # Return original if no match
        
        elif field_type == "integer":
            # Extract number from answer
            import re
            numbers = re.findall(r'\d+', answer)
            if numbers:
                return int(numbers[0])
            return None
        
        else:
            # Return as string for open-ended questions
            return answer.strip()
    
    async def stop_streaming_assessment(self) -> Dict[str, Any]:
        """Stop streaming assessment and generate final results."""
        if not self.current_session_id:
            return {"error": "No active session"}
        
        # Stop audio processing
        self.audio_processor.stop_streaming()
        
        # Generate final assessment report
        assessment_result = self._generate_assessment_report()
        
        # Save results
        result_file = self.output_dir / f"golf_assessment_{self.current_session_id}.json"
        with open(result_file, 'w') as f:
            json.dump(assessment_result, f, indent=2)
        
        self.logger.info(f"Golf assessment session {self.current_session_id} completed")
        
        # Reset session
        session_id = self.current_session_id
        self.current_session_id = None
        self.conversation_context = {}
        self.detected_qa_pairs = []
        
        return {
            "success": True,
            "session_id": session_id,
            "result_file": str(result_file),
            "assessment_result": assessment_result
        }
    
    def _generate_assessment_report(self) -> Dict[str, Any]:
        """Generate comprehensive assessment report."""
        assessment_template = self.assessment_templates.get(
            self.conversation_context["assessment_type"], {}
        )
        
        # Calculate completion statistics
        total_fields = 0
        completed_fields = 0
        sections_completion = {}
        
        for section_name, section_data in assessment_template.get("sections", {}).items():
            section_fields = list(section_data.get("fields", {}).keys())
            section_completed = 0
            
            for field_name in section_fields:
                total_fields += 1
                if field_name in self.conversation_context["current_responses"]:
                    completed_fields += 1
                    section_completed += 1
            
            sections_completion[section_name] = {
                "total_fields": len(section_fields),
                "completed_fields": section_completed,
                "completion_rate": section_completed / len(section_fields) if section_fields else 0
            }
        
        # Generate coaching recommendations
        coaching_recommendations = self._generate_coaching_recommendations()
        
        return {
            "assessment_metadata": {
                "session_id": self.conversation_context["session_id"],
                "athlete_id": self.conversation_context["athlete_id"],
                "coach_id": self.conversation_context["coach_id"],
                "assessment_type": self.conversation_context["assessment_type"],
                "start_time": self.conversation_context["start_time"],
                "end_time": datetime.now().isoformat(),
                "duration_minutes": (datetime.now() - datetime.fromisoformat(
                    self.conversation_context["start_time"]
                )).total_seconds() / 60,
                "session_notes": self.conversation_context.get("session_notes")
            },
            "assessment_responses": self.conversation_context["current_responses"],
            "completion_statistics": {
                "total_fields": total_fields,
                "completed_fields": completed_fields,
                "overall_completion_rate": completed_fields / total_fields if total_fields else 0,
                "sections_completion": sections_completion
            },
            "conversation_analysis": {
                "total_qa_pairs": len(self.detected_qa_pairs),
                "mapped_qa_pairs": len([qa for qa in self.detected_qa_pairs if qa.field_mapping]),
                "qa_pairs": self.conversation_context["qa_pairs"]
            },
            "coaching_recommendations": coaching_recommendations,
            "confidence_scores": {
                field_name: response.get("confidence", 0)
                for field_name, response in self.conversation_context["current_responses"].items()
            }
        }
    
    def _generate_coaching_recommendations(self) -> Dict[str, Any]:
        """Generate coaching recommendations based on assessment responses."""
        recommendations = {
            "priority_areas": [],
            "strength_areas": [],
            "practice_suggestions": [],
            "mental_game_focus": [],
            "physical_development": []
        }
        
        responses = self.conversation_context["current_responses"]
        
        # Analyze technical skills
        if "weakest_area" in responses:
            weak_area = responses["weakest_area"]["extracted_value"]
            if weak_area:
                recommendations["priority_areas"].append({
                    "area": "Technical Skills",
                    "focus": weak_area,
                    "recommendation": f"Focus practice sessions on {weak_area} improvement"
                })
        
        if "strongest_club" in responses:
            strong_club = responses["strongest_club"]["extracted_value"]
            if strong_club:
                recommendations["strength_areas"].append({
                    "area": "Technical Skills", 
                    "strength": strong_club,
                    "recommendation": f"Use confidence with {strong_club} to build overall game confidence"
                })
        
        # Analyze mental game
        if "pressure_handling" in responses:
            pressure_response = responses["pressure_handling"]["extracted_value"]
            if pressure_response and any(word in pressure_response.lower() for word in ["nervous", "struggle", "difficult"]):
                recommendations["mental_game_focus"].append({
                    "area": "Pressure Management",
                    "recommendation": "Develop mental resilience techniques and pressure situation practice"
                })
        
        if "pre_shot_routine" in responses:
            has_routine = responses["pre_shot_routine"]["extracted_value"]
            if not has_routine:
                recommendations["mental_game_focus"].append({
                    "area": "Pre-shot Routine",
                    "recommendation": "Establish and practice consistent pre-shot routine"
                })
        
        # Analyze physical fitness
        if "flexibility" in responses:
            flexibility = responses["flexibility"]["extracted_value"]
            if flexibility in ["Below Average", "Poor"]:
                recommendations["physical_development"].append({
                    "area": "Flexibility",
                    "recommendation": "Implement regular stretching and mobility routine"
                })
        
        if "fitness_routine" in responses:
            has_fitness = responses["fitness_routine"]["extracted_value"]
            if not has_fitness:
                recommendations["physical_development"].append({
                    "area": "Overall Fitness",
                    "recommendation": "Develop golf-specific fitness routine for improved performance"
                })
        
        # Practice suggestions based on goals
        if "primary_goal" in responses:
            goal = responses["primary_goal"]["extracted_value"]
            if goal:
                recommendations["practice_suggestions"].append({
                    "goal": goal,
                    "suggestion": f"Create structured practice plan aligned with goal: {goal}"
                })
        
        if "practice_time" in responses:
            practice_time = responses["practice_time"]["extracted_value"]
            if practice_time:
                recommendations["practice_suggestions"].append({
                    "time_available": practice_time,
                    "suggestion": f"Optimize practice efficiency for available time: {practice_time}"
                })
        
        return recommendations


# Demo Functions

def stream_audio_file(analyzer, audio_file_path: str, chunk_size_bytes: int = 32000, delay_seconds: float = 0.3):
    """
    Stream a real audio file to the analyzer in chunks.
    
    Args:
        analyzer: GolfAssessmentAnalyzer instance
        audio_file_path: Path to real audio file (WAV, MP3, etc.)
        chunk_size_bytes: Size of each chunk in bytes
        delay_seconds: Delay between chunks to simulate real-time streaming
    """
    audio_path = Path(audio_file_path)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_file_path}")
        return False
    
    print(f"📁 Streaming audio file: {audio_path.name}")
    print(f"   File size: {audio_path.stat().st_size:,} bytes")
    
    with open(audio_path, 'rb') as f:
        chunk_count = 0
        while True:
            chunk = f.read(chunk_size_bytes)
            if not chunk:
                break
            
            chunk_count += 1
            print(f"   📦 Sending chunk {chunk_count} ({len(chunk):,} bytes)")
            analyzer.add_audio_stream_chunk(chunk)
            
            # Simulate real-time streaming delay
            time.sleep(delay_seconds)
    
    print(f"✅ Finished streaming {chunk_count} chunks")
    return True

async def run_simulated_streaming_demo(audio_file_path: Optional[str] = None):
    """Run a streaming golf coaching assessment demo with real audio file."""
    print("🏌️ Golf Coaching Assessment - Streaming Demo")
    print("=" * 55)
    print()
    
    # Initialize analyzer
    analyzer = GolfAssessmentAnalyzer()
    
    # Start assessment session
    session_id = await analyzer.start_streaming_assessment(
        athlete_id="GOLFER_001",
        coach_id="COACH_001",
        session_notes="Assessment session with real audio streaming"
    )
    
    print(f"✅ Started streaming assessment session: {session_id}")
    print()
    
    if audio_file_path:
        # Stream real audio file
        print("🎙️ Streaming real audio file...")
        
        # Reduce buffer duration for faster processing during demo
        analyzer.audio_processor.buffer_duration = 10.0  # seconds
        analyzer.audio_processor.overlap_duration = 2.0
        
        success = stream_audio_file(analyzer, audio_file_path, chunk_size_bytes=32000, delay_seconds=0.5)
        
        if success:
            # Give some time for final processing
            print("⏳ Waiting for final audio processing...")
            await asyncio.sleep(5)
            
            # Force processing of any remaining buffer
            analyzer.audio_processor._trigger_processing()
            await asyncio.sleep(2)
        else:
            print("❌ Failed to stream audio file")
    else:
        # Fallback: demonstrate with sample conversation scenarios
        print("📝 No audio file provided - using simulated conversation data...")
        print("   (To use real audio, call: run_simulated_streaming_demo('/path/to/audio.wav'))")
        print()
        
        # Create simulated conversation for demo purposes
        conversation_scenarios = [
            "Coach: How long have you been playing golf? Athlete: I've been playing for about 8 years now.",
            "Coach: What's your current handicap? Athlete: I'm around a 12 handicap right now.", 
            "Coach: What club do you feel most confident with? Athlete: Definitely my 7-iron, I can always rely on it.",
            "Coach: What part of your game needs the most work? Athlete: My putting is really inconsistent, I struggle on the greens.",
            "Coach: How do you handle pressure situations? Athlete: I get pretty nervous, especially on important putts."
        ]
        
        print("🎙️ Processing conversation scenarios...")
        for i, scenario in enumerate(conversation_scenarios, 1):
            print(f"   Scenario {i}: {scenario[:50]}...")
            
            # Note: This creates placeholder data for demo - real implementation needs audio
            placeholder_data = f"scenario_{i}_placeholder".encode('utf-8')
            analyzer.add_audio_stream_chunk(placeholder_data)
            
            await asyncio.sleep(1)
    
    print()
    print("⏸️ Stopping streaming assessment...")
    
    # Stop assessment and get results
    final_result = await analyzer.stop_streaming_assessment()
    
    if final_result.get("success"):
        print("✅ Assessment completed successfully!")
        print()
        
        # Display results summary
        assessment = final_result["assessment_result"]
        completion_stats = assessment["completion_statistics"]
        
        print("📊 Assessment Results Summary:")
        print(f"   Overall Completion: {completion_stats['overall_completion_rate']:.1%}")
        print(f"   Fields Completed: {completion_stats['completed_fields']}/{completion_stats['total_fields']}")
        print()
        
        print("📝 Completed Assessment Fields:")
        for field_name, response in assessment["assessment_responses"].items():
            print(f"   • {field_name}: {response['extracted_value']}")
        print()
        
        print("🎯 Coaching Recommendations:")
        recommendations = assessment["coaching_recommendations"]
        
        if recommendations["priority_areas"]:
            print("   Priority Areas:")
            for priority in recommendations["priority_areas"]:
                print(f"     - {priority['recommendation']}")
        
        if recommendations["mental_game_focus"]:
            print("   Mental Game Focus:")
            for mental in recommendations["mental_game_focus"]:
                print(f"     - {mental['recommendation']}")
        
        print()
        print(f"📄 Detailed results saved to: {final_result['result_file']}")
    
    else:
        print("❌ Assessment failed:", final_result.get("error"))


def create_sample_golf_coaching_scenarios():
    """Create sample golf coaching conversation scenarios."""
    scenarios = {
        "beginner_assessment": {
            "athlete_profile": {
                "experience_level": "Beginner",
                "years_playing": 1,
                "typical_challenges": ["Basic swing mechanics", "Course management", "Equipment familiarity"]
            },
            "conversation_flow": [
                ("Coach", "Tell me about your golf experience. How long have you been playing?"),
                ("Athlete", "I just started about a year ago. I'm still learning the basics."),
                ("Coach", "What made you interested in taking up golf?"),
                ("Athlete", "My coworkers play a lot, and I wanted to join them for business rounds."),
                ("Coach", "Do you have your own set of clubs?"),
                ("Athlete", "Yes, I bought a starter set. I'm not sure if they're the right fit though."),
                ("Coach", "What's the most challenging part of the game for you right now?"),
                ("Athlete", "Consistency. Sometimes I hit it great, other times I completely miss the ball."),
                ("Coach", "How often are you able to practice or play?"),
                ("Athlete", "Maybe once or twice a month. I'd like to do more but work is busy.")
            ]
        },
        "intermediate_assessment": {
            "athlete_profile": {
                "experience_level": "Intermediate",
                "years_playing": 5,
                "handicap": 15,
                "typical_challenges": ["Course management", "Short game", "Mental pressure"]
            },
            "conversation_flow": [
                ("Coach", "What's your current handicap?"),
                ("Athlete", "I'm sitting at about a 15 right now, been stuck there for a while."),
                ("Coach", "What would you like to get it down to?"),
                ("Athlete", "Single digits would be amazing. Even getting to 10 would be great progress."),
                ("Coach", "What part of your game do you think is holding you back?"),
                ("Athlete", "Probably my short game. I can get near the green okay, but then I struggle with chipping and putting."),
                ("Coach", "How's your putting consistency?"),
                ("Athlete", "Not great. I three-putt way too often, especially from longer distances."),
                ("Coach", "Do you have a pre-shot routine?"),
                ("Athlete", "Sort of, but I don't stick to it when I'm nervous or under pressure."),
                ("Coach", "How do you typically handle pressure situations on the course?"),
                ("Athlete", "Not well. I tend to overthink and get in my own head.")
            ]
        },
        "advanced_assessment": {
            "athlete_profile": {
                "experience_level": "Advanced",
                "years_playing": 15,
                "handicap": 3,
                "typical_challenges": ["Fine-tuning", "Mental game", "Competition preparation"]
            },
            "conversation_flow": [
                ("Coach", "You're playing at a pretty high level. What brings you in for coaching?"),
                ("Athlete", "I want to get to scratch and maybe play some amateur tournaments."),
                ("Coach", "What's your current handicap?"),
                ("Athlete", "I'm a 3 handicap, but I feel like I should be better given how much I practice."),
                ("Coach", "Tell me about your practice routine."),
                ("Athlete", "I practice about 4-5 times a week, mix of range work and short game."),
                ("Coach", "Are you working on anything specific in your swing?"),
                ("Athlete", "My ball striking is pretty good, but I want more distance off the tee."),
                ("Coach", "How's your course management and strategy?"),
                ("Athlete", "I think that's where I lose strokes. I take too many risks sometimes."),
                ("Coach", "Have you played in competitions before?"),
                ("Athlete", "Club tournaments mostly. I'd like to try some regional amateur events.")
            ]
        }
    }
    
    return scenarios


async def run_scenario_based_demo(audio_files: Optional[Dict[str, str]] = None):
    """Run demo with realistic golf coaching scenarios using real audio files."""
    print("🏌️ Golf Coaching Assessment - Scenario-Based Demo")
    print("=" * 55)
    print()
    
    scenarios = create_sample_golf_coaching_scenarios()
    
    for scenario_name, scenario_data in scenarios.items():
        print(f"📋 Running Scenario: {scenario_name.replace('_', ' ').title()}")
        print(f"   Athlete Level: {scenario_data['athlete_profile']['experience_level']}")
        print()
        
        # Initialize analyzer for this scenario
        analyzer = GolfAssessmentAnalyzer()
        
        # Start assessment
        session_id = await analyzer.start_streaming_assessment(
            athlete_id=f"ATHLETE_{scenario_name.upper()}",
            coach_id="COACH_DEMO",
            session_notes=f"Assessment for {scenario_data['athlete_profile']['experience_level']} level athlete"
        )
        
        # Check if we have a real audio file for this scenario
        if audio_files and scenario_name in audio_files:
            audio_file_path = audio_files[scenario_name]
            print(f"🎙️ Processing real audio file: {Path(audio_file_path).name}")
            
            # Reduce buffer duration for demo
            analyzer.audio_processor.buffer_duration = 8.0
            analyzer.audio_processor.overlap_duration = 1.0
            
            success = stream_audio_file(analyzer, audio_file_path, chunk_size_bytes=24000, delay_seconds=0.3)
            
            if success:
                await asyncio.sleep(3)
                analyzer.audio_processor._trigger_processing()
                await asyncio.sleep(1)
        else:
            # Fallback to conversation flow demo
            print("🎙️ Processing conversation flow (no audio file provided)...")
            for speaker, dialogue in scenario_data["conversation_flow"]:
                print(f"   {speaker}: {dialogue}")
                
                # Create placeholder data (note: not real audio)
                dialogue_placeholder = f"{speaker}:{dialogue}".encode('utf-8')
                analyzer.add_audio_stream_chunk(dialogue_placeholder)
                
                await asyncio.sleep(0.5)
        
        print()
        
        # Complete assessment
        result = await analyzer.stop_streaming_assessment()
        
        if result.get("success"):
            assessment = result["assessment_result"]
            completion_rate = assessment["completion_statistics"]["overall_completion_rate"]
            
            print(f"✅ Scenario completed - {completion_rate:.1%} completion rate")
            print()
        
        print("-" * 55)
        print()


async def main():
    """Main demo function."""
    print("🏌️ Azure AI Content Understanding - Golf Coaching Assessment Demo")
    print("=" * 70)
    print()
    print("This demo shows how to use streaming audio processing to automatically")
    print("detect questions and answers in golf coaching conversations and fill out")
    print("comprehensive assessment forms.")
    print()
    
    # Example audio file paths (update these to your actual audio files)
    sample_audio_files = {
        "main_demo": r"C:\path\to\golf_coaching_session.wav",
        "beginner_assessment": r"C:\path\to\beginner_session.wav", 
        "intermediate_assessment": r"C:\path\to\intermediate_session.wav",
        "advanced_assessment": r"C:\path\to\advanced_session.wav"
    }
    
    try:
        print("🎵 Audio File Configuration:")
        print("   To use real audio files, update the 'sample_audio_files' paths in main()")
        print("   Supported formats: WAV, MP3, M4A, OGG, FLAC")
        print()
        
        # Run streaming demo with real audio (if file exists)
        main_audio_path = sample_audio_files.get("main_demo")
        if main_audio_path and Path(main_audio_path).exists():
            print("🎙️ Running demo with real audio file...")
            await run_simulated_streaming_demo(main_audio_path)
        else:
            print("📝 Running demo with simulated data (no audio file found)...")
            await run_simulated_streaming_demo()
        
        print("\n" + "=" * 70)
        print()
        
        # Run scenario-based demo 
        scenario_audio_files = {k: v for k, v in sample_audio_files.items() if k != "main_demo"}
        existing_audio_files = {k: v for k, v in scenario_audio_files.items() if Path(v).exists()}
        
        if existing_audio_files:
            print(f"🎵 Found {len(existing_audio_files)} scenario audio files")
            await run_scenario_based_demo(existing_audio_files)
        else:
            print("📝 Running scenario demo with simulated conversation data...")
            await run_scenario_based_demo()
        
        print("🎉 Golf coaching assessment demo completed successfully!")
        print()
        print("Key Features Demonstrated:")
        print("✅ Real-time audio stream processing")
        print("✅ Automatic question-answer detection")
        print("✅ Smart field mapping to assessment forms")
        print("✅ Coaching recommendations generation")
        print("✅ Multiple player skill level scenarios")
        print()
        print("The system can be integrated with:")
        print("• Live audio streaming from coaching sessions")
        print("• Golf lesson management systems")
        print("• Player development tracking platforms")
        print("• Performance analytics dashboards")
        print()
        print("💡 To test with your own audio files:")
        print("   1. Update the file paths in sample_audio_files")
        print("   2. Ensure audio files contain golf coaching conversations")
        print("   3. Use high-quality audio for best transcription results")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
