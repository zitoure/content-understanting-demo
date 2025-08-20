"""
Golf Coaching Analytics and Streaming Utilities

This module provides specialized utilities for golf coaching assessment,
including performance analytics, streaming audio processing, and player
development tracking.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class SkillLevel(Enum):
    """Golf skill level classifications."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class GolfMetric(Enum):
    """Golf performance metrics."""
    HANDICAP = "handicap"
    DRIVING_DISTANCE = "driving_distance"
    DRIVING_ACCURACY = "driving_accuracy"
    GREENS_IN_REGULATION = "greens_in_regulation"
    PUTTING_AVERAGE = "putting_average"
    SCRAMBLING = "scrambling"
    COURSE_MANAGEMENT = "course_management"


@dataclass
class PlayerProfile:
    """Comprehensive golf player profile."""
    player_id: str
    name: str
    age: Optional[int]
    years_playing: int
    current_handicap: Optional[float]
    skill_level: SkillLevel
    dominant_hand: str
    home_course: Optional[str]
    goals: List[str]
    last_assessment: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "player_id": self.player_id,
            "name": self.name,
            "age": self.age,
            "years_playing": self.years_playing,
            "current_handicap": self.current_handicap,
            "skill_level": self.skill_level.value,
            "dominant_hand": self.dominant_hand,
            "home_course": self.home_course,
            "goals": self.goals,
            "last_assessment": self.last_assessment.isoformat() if self.last_assessment else None
        }


class StreamingAudioAnalyzer:
    """
    Advanced streaming audio analyzer specifically designed for golf coaching sessions.
    Handles real-time audio processing, speaker diarization, and conversation flow analysis.
    """
    
    def __init__(self, buffer_size: int = 4096, sample_rate: int = 16000):
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.audio_buffer = bytearray()
        self.conversation_flow = []
        self.current_speaker = None
        self.speaker_change_threshold = 2.0  # seconds
        self.silence_threshold = 0.01
        self.min_speech_duration = 0.5  # seconds
    
    def process_audio_chunk(self, audio_chunk: bytes, timestamp: float) -> Dict[str, Any]:
        """
        Process incoming audio chunk and detect conversation patterns.
        
        Args:
            audio_chunk: Raw audio data
            timestamp: Timestamp of the audio chunk
            
        Returns:
            Analysis results including speaker detection and conversation flow
        """
        # Add to buffer
        self.audio_buffer.extend(audio_chunk)
        
        # Analyze audio properties (simplified - in real implementation use proper audio analysis)
        analysis_result = {
            "timestamp": timestamp,
            "chunk_size": len(audio_chunk),
            "buffer_size": len(self.audio_buffer),
            "estimated_speakers": self._estimate_speaker_count(),
            "speech_detected": self._detect_speech(audio_chunk),
            "silence_duration": self._estimate_silence_duration(),
            "conversation_flow": self._analyze_conversation_flow()
        }
        
        # Update conversation flow
        if analysis_result["speech_detected"]:
            self._update_conversation_flow(timestamp, analysis_result)
        
        return analysis_result
    
    def _estimate_speaker_count(self) -> int:
        """Estimate number of speakers in current buffer."""
        # Simplified speaker estimation
        # In real implementation, use advanced speaker diarization
        if len(self.audio_buffer) < 1000:
            return 1
        
        # Basic frequency analysis to estimate speakers
        # This is a placeholder - real implementation would use proper audio analysis
        return min(2, max(1, len(self.audio_buffer) // 10000))
    
    def _detect_speech(self, audio_chunk: bytes) -> bool:
        """Detect if audio chunk contains speech."""
        if len(audio_chunk) == 0:
            return False
        
        # Simple energy-based speech detection
        # In real implementation, use VAD (Voice Activity Detection)
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        energy = np.mean(np.abs(audio_array))
        
        return energy > self.silence_threshold * 32768  # Assuming 16-bit audio
    
    def _estimate_silence_duration(self) -> float:
        """Estimate duration of silence in current buffer."""
        # Simplified silence detection
        if len(self.audio_buffer) < 1000:
            return 0.0
        
        # Estimate based on buffer analysis
        return max(0.0, 3.0 - len(self.audio_buffer) / 10000)
    
    def _analyze_conversation_flow(self) -> Dict[str, Any]:
        """Analyze the conversation flow patterns."""
        if len(self.conversation_flow) < 2:
            return {"pattern": "insufficient_data"}
        
        recent_flow = self.conversation_flow[-10:]  # Last 10 interactions
        
        # Analyze turn-taking patterns
        speaker_changes = 0
        for i in range(1, len(recent_flow)):
            if recent_flow[i]["speaker"] != recent_flow[i-1]["speaker"]:
                speaker_changes += 1
        
        avg_turn_duration = np.mean([item["duration"] for item in recent_flow])
        
        return {
            "pattern": "normal_conversation",
            "speaker_changes": speaker_changes,
            "avg_turn_duration": avg_turn_duration,
            "interaction_rate": speaker_changes / len(recent_flow) if recent_flow else 0,
            "conversation_balance": self._calculate_conversation_balance(recent_flow)
        }
    
    def _update_conversation_flow(self, timestamp: float, analysis_result: Dict[str, Any]):
        """Update conversation flow tracking."""
        estimated_speaker = f"speaker_{analysis_result['estimated_speakers']}"
        
        # Detect speaker changes
        if (self.current_speaker != estimated_speaker or 
            len(self.conversation_flow) == 0 or
            timestamp - self.conversation_flow[-1]["timestamp"] > self.speaker_change_threshold):
            
            # Add new conversation segment
            self.conversation_flow.append({
                "timestamp": timestamp,
                "speaker": estimated_speaker,
                "duration": 0.0,
                "speech_detected": analysis_result["speech_detected"]
            })
            self.current_speaker = estimated_speaker
        
        # Update duration of current segment
        if self.conversation_flow:
            self.conversation_flow[-1]["duration"] = timestamp - self.conversation_flow[-1]["timestamp"]
    
    def _calculate_conversation_balance(self, flow_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate conversation balance between speakers."""
        speaker_times = {}
        
        for segment in flow_data:
            speaker = segment["speaker"]
            duration = segment["duration"]
            
            if speaker not in speaker_times:
                speaker_times[speaker] = 0.0
            speaker_times[speaker] += duration
        
        total_time = sum(speaker_times.values())
        
        if total_time == 0:
            return {}
        
        return {
            speaker: time / total_time 
            for speaker, time in speaker_times.items()
        }


class GolfPerformanceAnalyzer:
    """
    Analyzes golf performance data and generates insights for coaching.
    """
    
    def __init__(self):
        self.performance_metrics = {
            metric.value: [] for metric in GolfMetric
        }
        self.assessment_history = []
    
    def analyze_assessment_data(self, assessment_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze assessment data to generate performance insights.
        
        Args:
            assessment_result: Assessment results from golf coaching session
            
        Returns:
            Performance analysis and recommendations
        """
        responses = assessment_result.get("assessment_responses", {})
        
        # Extract performance indicators
        performance_indicators = self._extract_performance_indicators(responses)
        
        # Calculate skill assessment
        skill_assessment = self._assess_skill_level(responses)
        
        # Generate improvement priorities
        improvement_priorities = self._identify_improvement_priorities(responses)
        
        # Create practice plan
        practice_plan = self._generate_practice_plan(responses, improvement_priorities)
        
        # Benchmark against skill level
        benchmarks = self._benchmark_performance(responses, skill_assessment["estimated_level"])
        
        return {
            "performance_indicators": performance_indicators,
            "skill_assessment": skill_assessment,
            "improvement_priorities": improvement_priorities,
            "practice_plan": practice_plan,
            "benchmarks": benchmarks,
            "trend_analysis": self._analyze_trends(),
            "recommendations": self._generate_detailed_recommendations(
                performance_indicators, improvement_priorities
            )
        }
    
    def _extract_performance_indicators(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """Extract quantifiable performance indicators from responses."""
        indicators = {}
        
        # Extract handicap information
        if "current_handicap" in responses:
            handicap_str = responses["current_handicap"].get("extracted_value", "")
            handicap = self._parse_handicap(handicap_str)
            if handicap is not None:
                indicators["handicap"] = handicap
        
        # Extract frequency metrics
        if "rounds_per_month" in responses:
            rounds = responses["rounds_per_month"].get("extracted_value")
            if isinstance(rounds, int):
                indicators["play_frequency"] = rounds
                indicators["annual_rounds"] = rounds * 12
        
        # Extract experience level
        if "years_playing" in responses:
            years = responses["years_playing"].get("extracted_value")
            if isinstance(years, int):
                indicators["experience_years"] = years
        
        # Extract confidence levels
        confidence_metrics = {}
        for field_name, response in responses.items():
            if "confidence" in field_name or "consistent" in field_name:
                value = response.get("extracted_value")
                if value:
                    confidence_metrics[field_name] = self._normalize_confidence_score(value)
        
        if confidence_metrics:
            indicators["confidence_metrics"] = confidence_metrics
        
        return indicators
    
    def _assess_skill_level(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall skill level based on responses."""
        skill_indicators = []
        
        # Handicap-based assessment
        if "current_handicap" in responses:
            handicap = self._parse_handicap(responses["current_handicap"].get("extracted_value", ""))
            if handicap is not None:
                if handicap <= 5:
                    skill_indicators.append(("handicap", SkillLevel.ADVANCED))
                elif handicap <= 15:
                    skill_indicators.append(("handicap", SkillLevel.INTERMEDIATE))
                else:
                    skill_indicators.append(("handicap", SkillLevel.BEGINNER))
        
        # Experience-based assessment
        if "years_playing" in responses:
            years = responses["years_playing"].get("extracted_value")
            if isinstance(years, int):
                if years >= 10:
                    skill_indicators.append(("experience", SkillLevel.ADVANCED))
                elif years >= 3:
                    skill_indicators.append(("experience", SkillLevel.INTERMEDIATE))
                else:
                    skill_indicators.append(("experience", SkillLevel.BEGINNER))
        
        # Confidence-based assessment
        confidence_fields = ["driving_accuracy", "short_game_confidence", "putting_consistency"]
        high_confidence_count = 0
        
        for field in confidence_fields:
            if field in responses:
                value = responses[field].get("extracted_value", "")
                if value in ["Excellent", "Very Confident", "Very Consistent"]:
                    high_confidence_count += 1
        
        if high_confidence_count >= 2:
            skill_indicators.append(("confidence", SkillLevel.ADVANCED))
        elif high_confidence_count >= 1:
            skill_indicators.append(("confidence", SkillLevel.INTERMEDIATE))
        else:
            skill_indicators.append(("confidence", SkillLevel.BEGINNER))
        
        # Determine overall skill level
        skill_counts = {}
        for _, level in skill_indicators:
            skill_counts[level] = skill_counts.get(level, 0) + 1
        
        estimated_level = max(skill_counts, key=skill_counts.get) if skill_counts else SkillLevel.BEGINNER
        
        return {
            "estimated_level": estimated_level,
            "skill_indicators": skill_indicators,
            "confidence_score": len(skill_indicators) / 3.0,  # Normalize to 0-1
            "assessment_factors": list(skill_counts.keys())
        }
    
    def _identify_improvement_priorities(self, responses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify areas that need the most improvement."""
        priorities = []
        
        # Direct weakness identification
        if "weakest_area" in responses:
            weak_area = responses["weakest_area"].get("extracted_value", "")
            if weak_area:
                priorities.append({
                    "area": "Technical Skills",
                    "specific_focus": weak_area,
                    "priority_level": "High",
                    "reason": "Self-identified weakness",
                    "evidence": weak_area
                })
        
        # Low confidence areas
        low_confidence_areas = []
        confidence_fields = {
            "driving_accuracy": "Driving",
            "short_game_confidence": "Short Game", 
            "putting_consistency": "Putting"
        }
        
        for field, area_name in confidence_fields.items():
            if field in responses:
                value = responses[field].get("extracted_value", "")
                if value in ["Poor", "Not Confident", "Very Inconsistent", "Inconsistent"]:
                    low_confidence_areas.append(area_name)
        
        for area in low_confidence_areas:
            priorities.append({
                "area": area,
                "specific_focus": f"{area} fundamentals and consistency",
                "priority_level": "Medium",
                "reason": "Low confidence/consistency reported",
                "evidence": f"Self-reported low performance in {area}"
            })
        
        # Mental game issues
        if "pressure_handling" in responses:
            pressure_response = responses["pressure_handling"].get("extracted_value", "")
            if any(word in pressure_response.lower() for word in ["nervous", "struggle", "difficult", "bad"]):
                priorities.append({
                    "area": "Mental Game",
                    "specific_focus": "Pressure management and mental resilience",
                    "priority_level": "Medium",
                    "reason": "Difficulty handling pressure situations",
                    "evidence": pressure_response
                })
        
        if "pre_shot_routine" in responses:
            has_routine = responses["pre_shot_routine"].get("extracted_value")
            if not has_routine:
                priorities.append({
                    "area": "Mental Game", 
                    "specific_focus": "Pre-shot routine development",
                    "priority_level": "Medium",
                    "reason": "No consistent pre-shot routine",
                    "evidence": "No established routine"
                })
        
        # Physical limitations
        if "injuries_limitations" in responses:
            limitations = responses["injuries_limitations"].get("extracted_value", "")
            if limitations and limitations.lower() not in ["none", "no", "nothing"]:
                priorities.append({
                    "area": "Physical",
                    "specific_focus": "Address physical limitations",
                    "priority_level": "High",
                    "reason": "Physical limitations affecting performance",
                    "evidence": limitations
                })
        
        # Sort by priority level
        priority_order = {"High": 1, "Medium": 2, "Low": 3}
        priorities.sort(key=lambda x: priority_order.get(x["priority_level"], 3))
        
        return priorities
    
    def _generate_practice_plan(
        self, 
        responses: Dict[str, Any], 
        priorities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a structured practice plan based on assessment."""
        
        # Determine available practice time
        practice_time = responses.get("practice_time", {}).get("extracted_value", "")
        weekly_hours = self._parse_practice_time(practice_time)
        
        # Create practice plan structure
        practice_plan = {
            "weekly_hours": weekly_hours,
            "practice_sessions": [],
            "focus_distribution": {},
            "progression_milestones": []
        }
        
        if weekly_hours <= 2:
            # Limited time - focus on highest priority
            practice_plan["practice_sessions"] = [
                {
                    "session_type": "Focused Practice",
                    "duration": weekly_hours,
                    "frequency": "1x per week",
                    "focus_areas": [priorities[0]["area"]] if priorities else ["Short Game"],
                    "activities": self._get_practice_activities(priorities[0]["area"] if priorities else "Short Game")
                }
            ]
        elif weekly_hours <= 5:
            # Moderate time - split between priorities
            practice_plan["practice_sessions"] = [
                {
                    "session_type": "Technical Skills",
                    "duration": weekly_hours * 0.6,
                    "frequency": "2x per week",
                    "focus_areas": [p["area"] for p in priorities[:2]],
                    "activities": []
                },
                {
                    "session_type": "Course Play/Mental",
                    "duration": weekly_hours * 0.4,
                    "frequency": "1x per week",
                    "focus_areas": ["Mental Game", "Course Management"],
                    "activities": self._get_practice_activities("Mental Game")
                }
            ]
        else:
            # Ample time - comprehensive plan
            practice_plan["practice_sessions"] = [
                {
                    "session_type": "Technical Skills",
                    "duration": weekly_hours * 0.4,
                    "frequency": "2-3x per week",
                    "focus_areas": ["Driving", "Iron Play"],
                    "activities": self._get_practice_activities("Technical Skills")
                },
                {
                    "session_type": "Short Game",
                    "duration": weekly_hours * 0.3,
                    "frequency": "2x per week",
                    "focus_areas": ["Chipping", "Putting"],
                    "activities": self._get_practice_activities("Short Game")
                },
                {
                    "session_type": "Course Play",
                    "duration": weekly_hours * 0.3,
                    "frequency": "1-2x per week",
                    "focus_areas": ["Course Management", "Mental Game"],
                    "activities": self._get_practice_activities("Course Play")
                }
            ]
        
        # Add activities to sessions
        for session in practice_plan["practice_sessions"]:
            if not session["activities"]:
                session["activities"] = []
                for area in session["focus_areas"]:
                    session["activities"].extend(self._get_practice_activities(area))
        
        return practice_plan
    
    def _get_practice_activities(self, focus_area: str) -> List[str]:
        """Get specific practice activities for a focus area."""
        activities_map = {
            "Driving": [
                "Alignment stick drills",
                "Tempo practice with metronome",
                "Target practice at different distances"
            ],
            "Short Game": [
                "Chipping from various lies",
                "Distance control putting drills",
                "Bunker practice"
            ],
            "Putting": [
                "Distance control drills",
                "Breaking putt practice",
                "Pressure putting exercises"
            ],
            "Mental Game": [
                "Pre-shot routine practice",
                "Visualization exercises",
                "Pressure situation drills"
            ],
            "Technical Skills": [
                "Video analysis sessions",
                "Impact position drills",
                "Swing plane practice"
            ],
            "Course Play": [
                "Course management scenarios",
                "Shot selection practice",
                "Scoring strategy"
            ]
        }
        
        return activities_map.get(focus_area, ["General skill practice"])
    
    def _parse_handicap(self, handicap_str: str) -> Optional[float]:
        """Parse handicap from string description."""
        if not handicap_str:
            return None
        
        # Extract number from string
        import re
        numbers = re.findall(r'\d+\.?\d*', str(handicap_str))
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
        
        return None
    
    def _parse_practice_time(self, time_str: str) -> float:
        """Parse practice time from string description."""
        if not time_str:
            return 2.0  # Default
        
        time_lower = str(time_str).lower()
        
        # Extract hours
        import re
        hour_match = re.search(r'(\d+)\s*hour', time_lower)
        if hour_match:
            return float(hour_match.group(1))
        
        # Check for common phrases
        if any(phrase in time_lower for phrase in ["very little", "minimal", "not much"]):
            return 1.0
        elif any(phrase in time_lower for phrase in ["couple", "few"]):
            return 2.0
        elif any(phrase in time_lower for phrase in ["several", "multiple"]):
            return 4.0
        elif any(phrase in time_lower for phrase in ["lot", "many", "much"]):
            return 6.0
        
        return 3.0  # Default moderate
    
    def _normalize_confidence_score(self, confidence_value: str) -> float:
        """Convert confidence description to numerical score."""
        confidence_map = {
            "excellent": 1.0,
            "very confident": 0.9,
            "very consistent": 0.9,
            "good": 0.8,
            "confident": 0.7,
            "consistent": 0.7,
            "average": 0.5,
            "somewhat confident": 0.4,
            "inconsistent": 0.3,
            "not confident": 0.2,
            "very inconsistent": 0.1,
            "poor": 0.1
        }
        
        if not confidence_value:
            return 0.5
        
        confidence_lower = str(confidence_value).lower()
        for key, score in confidence_map.items():
            if key in confidence_lower:
                return score
        
        return 0.5  # Default
    
    def _benchmark_performance(self, responses: Dict[str, Any], skill_level: SkillLevel) -> Dict[str, Any]:
        """Benchmark performance against typical skill level standards."""
        benchmarks = {
            "skill_level": skill_level.value,
            "typical_ranges": {},
            "player_vs_typical": {},
            "improvement_potential": {}
        }
        
        # Define typical ranges by skill level
        typical_ranges = {
            SkillLevel.BEGINNER: {
                "handicap": (20, 36),
                "rounds_per_year": (5, 20),
                "practice_hours_per_week": (0, 2)
            },
            SkillLevel.INTERMEDIATE: {
                "handicap": (10, 20),
                "rounds_per_year": (15, 40),
                "practice_hours_per_week": (1, 4)
            },
            SkillLevel.ADVANCED: {
                "handicap": (0, 10),
                "rounds_per_year": (25, 60),
                "practice_hours_per_week": (3, 8)
            }
        }
        
        ranges = typical_ranges.get(skill_level, typical_ranges[SkillLevel.INTERMEDIATE])
        benchmarks["typical_ranges"] = ranges
        
        # Compare player to typical ranges
        if "current_handicap" in responses:
            player_handicap = self._parse_handicap(responses["current_handicap"].get("extracted_value", ""))
            if player_handicap is not None:
                typical_min, typical_max = ranges["handicap"]
                if player_handicap < typical_min:
                    benchmarks["player_vs_typical"]["handicap"] = "above_typical"
                elif player_handicap > typical_max:
                    benchmarks["player_vs_typical"]["handicap"] = "below_typical"
                else:
                    benchmarks["player_vs_typical"]["handicap"] = "typical"
        
        return benchmarks
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends from assessment history."""
        if len(self.assessment_history) < 2:
            return {"trend_analysis": "insufficient_data"}
        
        # In a real implementation, this would analyze historical data
        return {
            "trend_analysis": "available",
            "improvement_areas": [],
            "declining_areas": [],
            "stable_areas": []
        }
    
    def _generate_detailed_recommendations(
        self, 
        performance_indicators: Dict[str, Any],
        priorities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate detailed coaching recommendations."""
        recommendations = []
        
        # Priority-based recommendations
        for priority in priorities[:3]:  # Top 3 priorities
            recommendations.append({
                "category": priority["area"],
                "type": "improvement_focus",
                "recommendation": f"Focus on {priority['specific_focus']}",
                "rationale": priority["reason"],
                "timeline": "4-6 weeks",
                "success_metrics": self._get_success_metrics(priority["area"])
            })
        
        # Performance-based recommendations
        if "handicap" in performance_indicators:
            handicap = performance_indicators["handicap"]
            if handicap > 20:
                recommendations.append({
                    "category": "Overall Development",
                    "type": "structured_learning",
                    "recommendation": "Focus on fundamental skills development",
                    "rationale": "Higher handicap indicates need for basic skill building",
                    "timeline": "12-16 weeks",
                    "success_metrics": ["Reduce handicap by 3-5 strokes", "Improve consistency"]
                })
        
        return recommendations
    
    def _get_success_metrics(self, area: str) -> List[str]:
        """Get success metrics for specific improvement area."""
        metrics_map = {
            "Driving": ["Increase fairway percentage", "Improve distance consistency"],
            "Short Game": ["Reduce chips within 10 feet", "Improve up-and-down percentage"],
            "Putting": ["Reduce putts per round", "Improve distance control"],
            "Mental Game": ["Maintain focus throughout round", "Reduce penalty strokes"],
            "Technical Skills": ["Improve ball striking consistency", "Better impact position"],
            "Physical": ["Increase flexibility", "Reduce fatigue impact"]
        }
        
        return metrics_map.get(area, ["Improve overall performance"])


def create_comprehensive_golf_report(
    assessment_result: Dict[str, Any],
    performance_analysis: Dict[str, Any],
    player_profile: PlayerProfile
) -> Dict[str, Any]:
    """Create a comprehensive golf coaching report."""
    
    report = {
        "report_metadata": {
            "generated_date": datetime.now().isoformat(),
            "player_id": player_profile.player_id,
            "player_name": player_profile.name,
            "assessment_session": assessment_result.get("assessment_metadata", {}),
            "report_type": "comprehensive_golf_coaching_assessment"
        },
        "player_profile": player_profile.to_dict(),
        "assessment_summary": {
            "completion_rate": assessment_result.get("completion_statistics", {}).get("overall_completion_rate", 0),
            "key_findings": [],
            "overall_impression": ""
        },
        "performance_analysis": performance_analysis,
        "coaching_plan": {
            "immediate_focus": [],
            "short_term_goals": [],
            "long_term_development": [],
            "practice_schedule": performance_analysis.get("practice_plan", {})
        },
        "next_steps": {
            "recommended_actions": [],
            "follow_up_timeline": "2-4 weeks",
            "progress_tracking": []
        }
    }
    
    # Generate key findings
    priorities = performance_analysis.get("improvement_priorities", [])
    if priorities:
        report["assessment_summary"]["key_findings"] = [
            f"Primary focus area: {priorities[0]['area']} - {priorities[0]['specific_focus']}"
        ]
        
        if len(priorities) > 1:
            report["assessment_summary"]["key_findings"].append(
                f"Secondary focus: {priorities[1]['area']}"
            )
    
    # Generate overall impression
    skill_level = performance_analysis.get("skill_assessment", {}).get("estimated_level")
    if skill_level:
        report["assessment_summary"]["overall_impression"] = (
            f"Player demonstrates {skill_level.value} level skills with clear development opportunities."
        )
    
    # Generate coaching plan
    if priorities:
        report["coaching_plan"]["immediate_focus"] = [
            p["specific_focus"] for p in priorities[:2]
        ]
        
        report["coaching_plan"]["short_term_goals"] = [
            f"Improve {p['area'].lower()}" for p in priorities[:3]
        ]
    
    return report
