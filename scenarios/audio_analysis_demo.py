"""
Audio Analysis Demo for Azure AI Content Understanding

This script demonstrates how to analyze audio files using both prebuilt and custom audio analyzers.
It supports various audio formats and provides transcription, diarization, and custom field extraction.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import sys

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from content_understanding_client import AzureContentUnderstandingClient, create_client_from_env

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """
    Audio analyzer for Azure AI Content Understanding.
    Handles audio processing with both prebuilt and custom analyzers.
    """
    
    def __init__(self, client: AzureContentUnderstandingClient):
        self.client = client
        self.output_dir = Path(os.getenv("DEFAULT_OUTPUT_DIR", "./output"))
        self.output_dir.mkdir(exist_ok=True)
    
    def analyze_with_prebuilt_analyzer(
        self, 
        audio_url: str,
        analyzer_type: str = "general",
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze audio using prebuilt analyzers.
        
        Args:
            audio_url: URL of the audio file to analyze
            analyzer_type: Type of prebuilt analyzer ("general" or "call_center")
            output_file: Optional output file path for results
            
        Returns:
            Analysis results dictionary
        """
        # Select appropriate analyzer
        if analyzer_type == "call_center":
            analyzer_id = "prebuilt-callCenter"
            logger.info("Using prebuilt call center analyzer")
        else:
            analyzer_id = "prebuilt-audioAnalyzer"
            logger.info("Using prebuilt general audio analyzer")
        
        logger.info(f"Starting audio analysis with {analyzer_id}")
        logger.info(f"Audio URL: {audio_url}")
        
        # Start analysis
        analysis_request = self.client.analyze_content(
            analyzer_id=analyzer_id,
            content_url=audio_url
        )
        
        request_id = analysis_request["request_id"]
        logger.info(f"Analysis started with request ID: {request_id}")
        
        # Wait for completion (audio analysis can take longer)
        result = self.client.wait_for_analysis_completion(
            request_id, 
            max_wait_time=600,  # 10 minutes for audio
            poll_interval=10
        )
        
        # Save results if output file specified
        if output_file:
            output_path = self.output_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_path}")
        
        return result
    
    def create_custom_audio_analyzer(
        self, 
        analyzer_id: str,
        base_analyzer: str = "prebuilt-audioAnalyzer",
        description: str = "Custom audio analyzer",
        custom_fields: Optional[Dict[str, Any]] = None,
        locales: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a custom audio analyzer with specified fields and configuration.
        
        Args:
            analyzer_id: Unique identifier for the analyzer
            base_analyzer: Base analyzer to extend ("prebuilt-audioAnalyzer" or "prebuilt-callCenter")
            description: Description of the analyzer
            custom_fields: Custom field schema for extraction
            locales: List of language locales to support
            
        Returns:
            Analyzer creation result
        """
        existing_analyzer_ids = self.client.list_analyzer_ids()
        if analyzer_id in existing_analyzer_ids:
            logger.warning(f"Analyzer with ID '{analyzer_id}' already exists. Reusing it")
            return {"status": "exists", "analyzer_id": analyzer_id}

        # Default custom fields for audio analysis
        if custom_fields is None:
            if base_analyzer == "prebuilt-callCenter":
                custom_fields = {
                    "CallSummary": {
                        "type": "string",
                        "method": "generate",
                        "description": "Comprehensive summary of the call"
                    },
                    "CustomerSentiment": {
                        "type": "string",
                        "enum": ["Positive", "Neutral", "Negative", "Mixed"],
                        "description": "Overall sentiment of the customer"
                    },
                    "AgentPerformance": {
                        "type": "string",
                        "enum": ["Excellent", "Good", "Average", "Poor"],
                        "description": "Assessment of agent performance"
                    },
                    "KeyIssues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Main issues discussed in the call"
                    },
                    "ResolutionStatus": {
                        "type": "string",
                        "enum": ["Resolved", "Partially Resolved", "Unresolved", "Escalated"],
                        "description": "Status of issue resolution"
                    },
                    "FollowUpRequired": {
                        "type": "boolean",
                        "description": "Whether follow-up action is needed"
                    },
                    "CallDuration": {
                        "type": "string",
                        "description": "Duration of the call"
                    }
                }
            else:
                custom_fields = {
                    "Summary": {
                        "type": "string",
                        "method": "generate",
                        "description": "Summary of the audio content"
                    },
                    "MainTopics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Main topics discussed in the audio"
                    },
                    "SpeakerCount": {
                        "type": "integer",
                        "description": "Number of speakers identified"
                    },
                    "Sentiment": {
                        "type": "string",
                        "enum": ["Positive", "Neutral", "Negative"],
                        "description": "Overall sentiment of the conversation"
                    },
                    "ActionItems": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Action items mentioned in the audio"
                    },
                    "KeyQuotes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Important quotes from the conversation"
                    }
                }
        
        # Default locales
        if locales is None:
            locales = ["en-US", "es-ES", "fr-FR"]
        
        analyzer_config = {
            "description": description,
            "baseAnalyzerId": base_analyzer,
            "config": {
                "locales": locales,
                "returnDetails": True,
                "disableContentFiltering": False
            },
            "fieldSchema": {
                "fields": custom_fields
            }
        }
        
        logger.info(f"Creating custom audio analyzer: {analyzer_id}")
        creation_result = self.client.create_analyzer(analyzer_id, analyzer_config)
        
        # Wait for analyzer creation to complete
        if creation_result.get("operation_location"):
            logger.info("Waiting for analyzer creation to complete...")
            operation_url = creation_result["operation_location"]
            
            import time
            max_wait = 120  # 2 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                status_result = self.client.get_analyzer_operation_status(operation_url)
                status = status_result.get("status", "Unknown")
                
                if status.lower() == "succeeded":
                    logger.info(f"Analyzer {analyzer_id} created successfully")
                    break
                elif status.lower() in ["failed", "cancelled"]:
                    raise Exception(f"Analyzer creation failed: {status}")
                
                time.sleep(5)
            else:
                raise Exception("Analyzer creation timed out")
        
        return creation_result
    
    def analyze_with_custom_analyzer(
        self,
        analyzer_id: str,
        audio_url: str,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze audio using a custom analyzer.
        
        Args:
            analyzer_id: ID of the custom analyzer to use
            audio_url: URL of the audio file to analyze
            output_file: Optional output file path for results
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Analyzing audio with custom analyzer: {analyzer_id}")
        logger.info(f"Audio URL: {audio_url}")
        
        # Start analysis
        analysis_request = self.client.analyze_content(
            analyzer_id=analyzer_id,
            content_url=audio_url
        )
        
        request_id = analysis_request["request_id"]
        logger.info(f"Analysis started with request ID: {request_id}")
        
        # Wait for completion (audio analysis can take longer)
        result = self.client.wait_for_analysis_completion(
            request_id,
            max_wait_time=600,  # 10 minutes for audio
            poll_interval=10
        )
        
        # Save results if output file specified
        if output_file:
            output_path = self.output_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_path}")
        
        return result
    
    def extract_audio_insights(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key insights from audio analysis results.
        
        Args:
            result: Raw analysis results
            
        Returns:
            Simplified insights dictionary
        """
        insights = {
            "status": result.get("status"),
            "analyzer_id": result.get("result", {}).get("analyzerId"),
            "created_at": result.get("result", {}).get("createdAt"),
            "transcript": "",
            "speakers": [],
            "duration": {},
            "extracted_fields": {},
            "warnings": result.get("result", {}).get("warnings", [])
        }
        
        contents = result.get("result", {}).get("contents", [])
        if contents:
            content = contents[0]  # Get first content item
            
            # Extract transcript from markdown
            markdown = content.get("markdown", "")
            if "Transcript" in markdown:
                # Extract WEBVTT transcript section
                transcript_start = markdown.find("```\nWEBVTT")
                if transcript_start != -1:
                    transcript_end = markdown.find("```", transcript_start + 3)
                    if transcript_end != -1:
                        insights["transcript"] = markdown[transcript_start:transcript_end + 3]
            
            # Extract duration information
            if "startTimeMs" in content and "endTimeMs" in content:
                start_ms = content["startTimeMs"]
                end_ms = content["endTimeMs"]
                duration_ms = end_ms - start_ms
                
                insights["duration"] = {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": duration_ms,
                    "duration_seconds": duration_ms / 1000,
                    "duration_minutes": duration_ms / 60000
                }
            
            # Extract speaker information from transcript phrases
            transcript_phrases = content.get("transcriptPhrases", [])
            speakers = set()
            for phrase in transcript_phrases:
                if "speaker" in phrase:
                    speakers.add(phrase["speaker"])
            insights["speakers"] = list(speakers)
            
            # Extract custom fields
            fields = content.get("fields", {})
            for field_name, field_data in fields.items():
                field_type = field_data.get("type")
                if field_type == "string":
                    insights["extracted_fields"][field_name] = field_data.get("valueString")
                elif field_type == "array":
                    insights["extracted_fields"][field_name] = field_data.get("valueArray", [])
                elif field_type == "boolean":
                    insights["extracted_fields"][field_name] = field_data.get("valueBoolean")
                elif field_type == "integer":
                    insights["extracted_fields"][field_name] = field_data.get("valueInteger")
        
        return insights
    
    def generate_transcript_summary(self, transcript: str, max_length: int = 500) -> str:
        """
        Generate a simple summary from transcript text.
        
        Args:
            transcript: WEBVTT transcript text
            max_length: Maximum length of summary
            
        Returns:
            Summary text
        """
        # Simple extraction of spoken text from WEBVTT
        lines = transcript.split('\n')
        spoken_text = []
        
        for line in lines:
            line = line.strip()
            # Skip WEBVTT headers, timestamps, and empty lines
            if (line and 
                not line.startswith('WEBVTT') and 
                '-->' not in line and 
                not line.startswith('<v ') and
                not line.startswith('NOTE')):
                # Clean speaker tags
                if line.startswith('<v '):
                    if '>' in line:
                        line = line[line.index('>') + 1:]
                spoken_text.append(line)
        
        full_text = ' '.join(spoken_text)
        
        if len(full_text) <= max_length:
            return full_text
        else:
            return full_text[:max_length] + "..."


def main():
    """Main demo function for audio analysis."""
    # Load environment variables
    load_dotenv()
    
    # Create client
    try:
        client = create_client_from_env()
        analyzer = AudioAnalyzer(client)
        
        print("=== Azure AI Content Understanding - Audio Analysis Demo ===\n")
        
        # Sample audio URL (you can replace with your own)
        sample_audio_url = "https://github.com/Azure-Samples/azure-ai-content-understanding-python/raw/refs/heads/main/data/audio.wav"
        
        # Demo 1: Analyze with prebuilt general audio analyzer
        print("1. Analyzing audio with prebuilt general analyzer...")
        try:
            result = analyzer.analyze_with_prebuilt_analyzer(
                audio_url=sample_audio_url,
                analyzer_type="general",
                output_file="prebuilt_audio_analysis.json"
            )
            
            # Extract and display insights
            insights = analyzer.extract_audio_insights(result)
            print(f"   Status: {insights['status']}")
            print(f"   Analyzer: {insights['analyzer_id']}")
            print(f"   Duration: {insights['duration'].get('duration_minutes', 0):.1f} minutes")
            print(f"   Speakers detected: {len(insights['speakers'])}")
            
            # Show transcript preview
            if insights['transcript']:
                summary = analyzer.generate_transcript_summary(insights['transcript'], 200)
                print(f"   Transcript preview: {summary}")
            
            print("   ✓ General audio analysis completed successfully\n")
            
        except Exception as e:
            print(f"   ✗ General audio analysis failed: {e}\n")
        
        # Demo 2: Analyze with prebuilt call center analyzer
        print("2. Analyzing audio with prebuilt call center analyzer...")
        try:
            result = analyzer.analyze_with_prebuilt_analyzer(
                audio_url=sample_audio_url,
                analyzer_type="call_center",
                output_file="prebuilt_callcenter_analysis.json"
            )
            
            # Extract and display call center specific insights
            insights = analyzer.extract_audio_insights(result)
            print(f"   Status: {insights['status']}")
            print(f"   Call center analysis completed")
            print(f"   Duration: {insights['duration'].get('duration_minutes', 0):.1f} minutes")
            
            # Show extracted call center fields
            if insights['extracted_fields']:
                print("   Call center insights:")
                for field_name, field_value in insights['extracted_fields'].items():
                    if isinstance(field_value, list):
                        print(f"     {field_name}: {', '.join(field_value[:3])}{'...' if len(field_value) > 3 else ''}")
                    else:
                        value_str = str(field_value)[:100]
                        print(f"     {field_name}: {value_str}{'...' if len(str(field_value)) > 100 else ''}")
            
            print("   ✓ Call center analysis completed successfully\n")
            
        except Exception as e:
            print(f"   ✗ Call center analysis failed: {e}\n")
        
        # Demo 3: Create and use custom audio analyzer
        print("3. Creating custom audio analyzer...")
        custom_analyzer_id = "demo-meeting-analyzer"
        
        try:
            # Create custom analyzer for meeting analysis
            creation_result = analyzer.create_custom_audio_analyzer(
                analyzer_id=custom_analyzer_id,
                base_analyzer="prebuilt-audioAnalyzer",
                description="Demo meeting analyzer with custom fields",
                locales=["en-US"]
            )
            print(f"   ✓ Custom analyzer '{custom_analyzer_id}' created successfully\n")
            
            # Analyze with custom analyzer
            print("4. Analyzing audio with custom analyzer...")
            custom_result = analyzer.analyze_with_custom_analyzer(
                analyzer_id=custom_analyzer_id,
                audio_url=sample_audio_url,
                output_file="custom_audio_analysis.json"
            )
            
            # Extract and display custom analysis results
            insights = analyzer.extract_audio_insights(custom_result)
            print(f"   Status: {insights['status']}")
            print("   Custom extracted fields:")
            for field_name, field_value in insights['extracted_fields'].items():
                if isinstance(field_value, list):
                    print(f"     {field_name}: {', '.join(field_value[:2])}{'...' if len(field_value) > 2 else ''}")
                else:
                    value_str = str(field_value)[:80]
                    print(f"     {field_name}: {value_str}{'...' if len(str(field_value)) > 80 else ''}")
            
            print("   ✓ Custom audio analysis completed successfully\n")
            
        except Exception as e:
            print(f"   ✗ Custom audio analyzer demo failed: {e}\n")
        
        print("=== Audio Analysis Demo Completed ===")
        print(f"Results saved to: {analyzer.output_dir}")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Please check your configuration in .env file")


if __name__ == "__main__":
    main()
