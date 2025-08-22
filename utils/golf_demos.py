"""
Demo scenarios and helper functions for golf coaching assessment.
"""

import asyncio
import time
from pathlib import Path
from typing import Optional, Dict
from utils.golf_analyzer import GolfAssessmentAnalyzer


def process_audio_file(analyzer: GolfAssessmentAnalyzer, audio_file_path: str) -> bool:
    """
    Process an audio file with the analyzer.
    
    Args:
        analyzer: GolfAssessmentAnalyzer instance
        audio_file_path: Path to audio file (WAV, MP3, etc.)
        
    Returns:
        True if processing completed successfully, False otherwise
    """
    audio_path = Path(audio_file_path)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_file_path}")
        return False
    
    print(f"📁 Processing audio file: {audio_path.name}")
    print(f"   File size: {audio_path.stat().st_size:,} bytes")
    
    # Process the file (this would be handled by the analyzer's analyze_audio_file method)
    print(f"✅ Audio file processing initiated")
    return True


async def run_audio_analysis_demo(audio_file_path: Optional[str] = None):
    """Run a golf coaching assessment demo with audio file analysis."""
    print("🏌️ Golf Coaching Assessment - Audio Analysis Demo")
    print("=" * 55)
    print()
    
    # Initialize analyzer
    analyzer = GolfAssessmentAnalyzer()
    
    # Start assessment session
    session_id = await analyzer.start_assessment(
        athlete_id="GOLFER_001",
        coach_id="COACH_001",
        session_notes="Assessment session with audio file analysis"
    )
    
    print(f"✅ Started assessment session: {session_id}")
    print()
    
    if audio_file_path:
        # Process real audio file
        print("🎙️ Analyzing audio file...")
        
        analysis_result = await analyzer.analyze_audio_file(audio_file_path)
        
        if analysis_result.get("success"):
            print("✅ Audio analysis completed!")
            print(f"   Q&A pairs found: {analysis_result.get('qa_pairs_found', 0)}")
        else:
            print(f"❌ Failed to analyze audio file: {analysis_result.get('error')}")
    else:
        # Fallback: demonstrate with sample conversation scenarios
        print("📝 No audio file provided - using simulated conversation data...")
        print("   (To use real audio, call: run_audio_analysis_demo('/path/to/audio.wav'))")
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
            await asyncio.sleep(1)
    
    print()
    print("⏸️ Stopping assessment...")
    
    # Stop assessment and get results
    final_result = await analyzer.stop_assessment()
    
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
    final_result = await analyzer.stop_assessment()
    
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
        session_id = await analyzer.start_assessment(
            athlete_id=f"ATHLETE_{scenario_name.upper()}",
            coach_id="COACH_DEMO",
            session_notes=f"Assessment for {scenario_data['athlete_profile']['experience_level']} level athlete"
        )
        
        # Check if we have a real audio file for this scenario
        if audio_files and scenario_name in audio_files:
            audio_file_path = audio_files[scenario_name]
            print(f"🎙️ Processing real audio file: {Path(audio_file_path).name}")
            
            analysis_result = await analyzer.analyze_audio_file(audio_file_path)
            
            if analysis_result.get("success"):
                print("✅ Audio analysis completed!")
                print(f"   Q&A pairs found: {analysis_result.get('qa_pairs_found', 0)}")
            else:
                print(f"❌ Failed to analyze audio file: {analysis_result.get('error')}")
        else:
            # Fallback to conversation flow demo
            print("🎙️ Processing conversation flow (no audio file provided)...")
            for speaker, dialogue in scenario_data["conversation_flow"]:
                print(f"   {speaker}: {dialogue}")
                await asyncio.sleep(0.5)
                
                await asyncio.sleep(0.5)
        
        print()
        
        # Complete assessment
        result = await analyzer.stop_assessment()
        
        if result.get("success"):
            assessment = result["assessment_result"]
            completion_rate = assessment["completion_statistics"]["overall_completion_rate"]
            
            print(f"✅ Scenario completed - {completion_rate:.1%} completion rate")
            print()
        
        print("-" * 55)
        print()
