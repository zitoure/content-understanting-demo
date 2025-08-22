"""
Golf assessment data models and enums.
"""

import re
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime


class AssessmentSection(Enum):
    """Assessment sections for organizing fields."""
    BASIC_INFO = "basic_info"
    TECHNICAL_SKILLS = "technical_skills"
    MENTAL_GAME = "mental_game"
    PHYSICAL_FITNESS = "physical_fitness"
    GOALS_PLANNING = "goals_planning"


@dataclass
class QuestionAnswerPair:
    """Represents a detected question-answer pair from conversation."""
    question: str
    answer: str
    timestamp: float
    speaker_question: str = "coach"  # Speaker who asked the question
    speaker_answer: str = "athlete"  # Speaker who provided the answer
    field_extracted: float = 1.0  # Extraction status (1.0 = extracted, 0.0 = not extracted)
    section: Optional[AssessmentSection] = None
    field_mapping: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.8  # Confidence in the detection
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure field_extracted is a valid value
        if self.field_extracted not in [0.0, 1.0]:
            self.field_extracted = 1.0 if self.field_extracted > 0.5 else 0.0


class GolfAssessmentTemplates:
    """Golf assessment templates and field configurations."""
    
    @staticmethod
    def create_comprehensive_template() -> Dict[str, Any]:
        """Create comprehensive golf assessment template."""
        return {
            "name": "Comprehensive Golf Assessment",
            "description": "Complete assessment covering all aspects of golf performance",
            "sections": {
                "basic_info": {
                    "name": "Basic Information",
                    "description": "Fundamental player information",
                    "fields": {
                        "years_playing": {
                            "question": "How long have you been playing golf?",
                            "type": "string",
                            "question_type": "open_ended",
                            "keywords": ["years", "playing", "started", "began", "long", "time"],
                            "description": "Number of years the athlete has been playing golf"
                        },
                        "current_handicap": {
                            "question": "What is your current handicap?",
                            "type": "string",
                            "question_type": "open_ended",
                            "keywords": ["handicap", "index", "current", "what"],
                            "description": "Player's official handicap index"
                        },
                        "rounds_per_month": {
                            "question": "How many rounds do you play per month?",
                            "type": "integer",
                            "question_type": "open_ended",
                            "keywords": ["rounds", "play", "month", "often", "frequency"],
                            "description": "Average number of rounds played monthly"
                        },
                        "practice_frequency": {
                            "question": "How often do you practice?",
                            "type": "string",
                            "question_type": "single_choice",
                            "enum": ["Daily", "Several times a week", "Weekly", "Occasionally", "Rarely"],
                            "keywords": ["practice", "often", "frequency", "train"],
                            "description": "How frequently the player practices"
                        }
                    }
                },
                "technical_skills": {
                    "name": "Technical Skills",
                    "description": "Golf technique and skill assessment",
                    "fields": {
                        "strongest_club": {
                            "question": "What club do you feel most confident with?",
                            "type": "string",
                            "question_type": "open_ended",
                            "keywords": ["confident", "strongest", "best", "favorite", "club"],
                            "description": "Club the player feels most confident using"
                        },
                        "weakest_area": {
                            "question": "What area of your game needs the most work?",
                            "type": "string",
                            "question_type": "single_choice",
                            "enum": ["Driving", "Iron play", "Short game", "Putting", "Course management"],
                            "keywords": ["weakest", "work", "improve", "struggle", "difficult"],
                            "description": "Area of the game that needs the most improvement"
                        },
                        "driving_distance": {
                            "question": "What's your average driving distance?",
                            "type": "integer",
                            "question_type": "open_ended",
                            "keywords": ["driving", "distance", "yards", "far", "hit"],
                            "description": "Average driving distance in yards"
                        },
                        "ball_striking_consistency": {
                            "question": "How would you rate your ball striking consistency?",
                            "type": "string",
                            "question_type": "single_choice",
                            "enum": ["Excellent", "Good", "Average", "Below Average", "Poor"],
                            "keywords": ["ball", "striking", "consistency", "contact", "solid"],
                            "description": "Self-assessment of ball striking consistency"
                        }
                    }
                },
                "mental_game": {
                    "name": "Mental Game",
                    "description": "Mental aspects of golf performance",
                    "fields": {
                        "pressure_handling": {
                            "question": "How do you handle pressure situations?",
                            "type": "string",
                            "question_type": "open_ended",
                            "keywords": ["pressure", "nervous", "stress", "handle", "cope"],
                            "description": "How the player handles pressure situations"
                        },
                        "pre_shot_routine": {
                            "question": "Do you have a consistent pre-shot routine?",
                            "type": "boolean",
                            "question_type": "yes_no",
                            "keywords": ["routine", "pre-shot", "consistent", "same", "ritual"],
                            "description": "Whether player has a consistent pre-shot routine"
                        },
                        "course_management": {
                            "question": "How would you rate your course management skills?",
                            "type": "string",
                            "question_type": "single_choice",
                            "enum": ["Excellent", "Good", "Average", "Below Average", "Poor"],
                            "keywords": ["course", "management", "strategy", "smart", "decisions"],
                            "description": "Self-assessment of course management abilities"
                        },
                        "mental_toughness": {
                            "question": "How do you bounce back from bad shots?",
                            "type": "string",
                            "question_type": "open_ended",
                            "keywords": ["bounce", "back", "recover", "bad", "shots", "mistakes"],
                            "description": "Player's mental resilience and recovery from mistakes"
                        }
                    }
                },
                "physical_fitness": {
                    "name": "Physical Fitness",
                    "description": "Physical conditioning and fitness assessment",
                    "fields": {
                        "flexibility": {
                            "question": "How would you rate your flexibility?",
                            "type": "string",
                            "question_type": "single_choice",
                            "enum": ["Excellent", "Good", "Average", "Below Average", "Poor"],
                            "keywords": ["flexibility", "flexible", "stretch", "mobility"],
                            "description": "Self-assessment of flexibility and mobility"
                        },
                        "fitness_routine": {
                            "question": "Do you have a regular fitness routine?",
                            "type": "boolean",
                            "question_type": "yes_no",
                            "keywords": ["fitness", "routine", "exercise", "workout", "gym"],
                            "description": "Whether player maintains a regular fitness routine"
                        },
                        "stamina": {
                            "question": "How is your stamina during a full round?",
                            "type": "string",
                            "question_type": "single_choice",
                            "enum": ["Excellent", "Good", "Average", "Below Average", "Poor"],
                            "keywords": ["stamina", "endurance", "tired", "energy", "full round"],
                            "description": "Physical endurance during a complete round of golf"
                        },
                        "injury_history": {
                            "question": "Do you have any golf-related injuries or physical limitations?",
                            "type": "string",
                            "question_type": "open_ended",
                            "keywords": ["injury", "injuries", "pain", "limitations", "physical"],
                            "description": "Any physical limitations or injury history"
                        }
                    }
                },
                "goals_planning": {
                    "name": "Goals and Planning",
                    "description": "Golf goals and improvement planning",
                    "fields": {
                        "primary_goal": {
                            "question": "What is your primary golf goal?",
                            "type": "string",
                            "question_type": "open_ended",
                            "keywords": ["goal", "want", "achieve", "improve", "primary"],
                            "description": "Player's main golf improvement goal"
                        },
                        "target_handicap": {
                            "question": "What handicap would you like to achieve?",
                            "type": "string",
                            "question_type": "open_ended",
                            "keywords": ["target", "achieve", "goal", "handicap", "want"],
                            "description": "Target handicap the player wants to reach"
                        },
                        "practice_time": {
                            "question": "How much time can you dedicate to practice weekly?",
                            "type": "string",
                            "question_type": "open_ended",
                            "keywords": ["practice", "time", "weekly", "hours", "dedicate"],
                            "description": "Available practice time per week"
                        },
                        "lesson_frequency": {
                            "question": "How often would you like to have lessons?",
                            "type": "string",
                            "question_type": "single_choice",
                            "enum": ["Weekly", "Bi-weekly", "Monthly", "Quarterly", "As needed"],
                            "keywords": ["lessons", "coaching", "instruction", "often", "frequency"],
                            "description": "Desired frequency of golf lessons"
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def create_beginner_template() -> Dict[str, Any]:
        """Create beginner-focused golf assessment template."""
        return {
            "name": "Beginner Golf Assessment",
            "description": "Assessment focused on beginning golfers",
            "sections": {
                "basic_info": {
                    "name": "Basic Information",
                    "description": "Basic player information for beginners",
                    "fields": {
                        "years_playing": {
                            "question": "How long have you been playing golf?",
                            "type": "string",
                            "question_type": "open_ended",
                            "keywords": ["years", "playing", "started", "new"],
                            "description": "Experience level in golf"
                        },
                        "lesson_experience": {
                            "question": "Have you taken golf lessons before?",
                            "type": "boolean",
                            "question_type": "yes_no",
                            "keywords": ["lessons", "instruction", "taught", "coach"],
                            "description": "Previous lesson experience"
                        },
                        "equipment_owned": {
                            "question": "Do you have your own golf clubs?",
                            "type": "boolean",
                            "question_type": "yes_no",
                            "keywords": ["clubs", "equipment", "own", "have"],
                            "description": "Whether player owns golf equipment"
                        }
                    }
                },
                "learning_goals": {
                    "name": "Learning Goals",
                    "description": "What the beginner wants to learn",
                    "fields": {
                        "main_challenge": {
                            "question": "What's the most challenging part of golf for you?",
                            "type": "string",
                            "question_type": "open_ended",
                            "keywords": ["challenging", "difficult", "hard", "struggle"],
                            "description": "Biggest challenge faced by the beginner"
                        },
                        "priority_skill": {
                            "question": "What skill would you like to focus on first?",
                            "type": "string",
                            "question_type": "single_choice",
                            "enum": ["Basic swing", "Putting", "Chipping", "Rules", "Course etiquette"],
                            "keywords": ["focus", "first", "priority", "learn"],
                            "description": "Priority skill for initial focus"
                        }
                    }
                }
            }
        }


class FieldMatcher:
    """Utility class for matching questions to assessment fields."""
    
    @staticmethod
    def calculate_field_match_score(question: str, field_config: Dict[str, Any]) -> float:
        """
        Calculate how well a question matches a field configuration.
        
        Args:
            question: The question text to match
            field_config: Field configuration with keywords and patterns
            
        Returns:
            Match score between 0.0 and 1.0
        """
        if not question or not field_config:
            return 0.0
        
        question_lower = question.lower()
        keywords = field_config.get("keywords", [])
        
        if not keywords:
            return 0.0
        
        # Calculate keyword matches
        keyword_matches = 0
        for keyword in keywords:
            if keyword.lower() in question_lower:
                keyword_matches += 1
        
        # Base score from keyword matching
        base_score = keyword_matches / len(keywords)
        
        # Bonus for exact question match
        expected_question = field_config.get("question", "").lower()
        if expected_question and expected_question in question_lower:
            base_score += 0.3
        
        # Bonus for question type indicators
        question_type = field_config.get("question_type", "")
        if question_type == "yes_no":
            yes_no_indicators = ["do you", "are you", "have you", "can you", "will you"]
            if any(indicator in question_lower for indicator in yes_no_indicators):
                base_score += 0.2
        
        # Cap at 1.0
        return min(base_score, 1.0)
    
    @staticmethod
    def extract_numeric_value(answer: str) -> Optional[int]:
        """Extract numeric value from answer text."""
        numbers = re.findall(r'\d+', answer)
        return int(numbers[0]) if numbers else None
    
    @staticmethod
    def extract_boolean_value(answer: str) -> Optional[bool]:
        """Extract boolean value from answer text."""
        answer_lower = answer.lower()
        yes_indicators = ["yes", "yeah", "yep", "sure", "definitely", "absolutely", "correct", "true"]
        no_indicators = ["no", "nope", "not", "never", "negative", "false"]
        
        for indicator in yes_indicators:
            if indicator in answer_lower:
                return True
        
        for indicator in no_indicators:
            if indicator in answer_lower:
                return False
        
        return None
    
    @staticmethod
    def extract_enum_value(answer: str, enum_values: List[str]) -> Optional[str]:
        """Extract enum value from answer text."""
        answer_lower = answer.lower()
        
        # Exact match first
        for enum_value in enum_values:
            if enum_value.lower() in answer_lower:
                return enum_value
        
        # Partial matching
        for enum_value in enum_values:
            enum_words = enum_value.lower().split()
            if any(word in answer_lower for word in enum_words):
                return enum_value
        
        return None
