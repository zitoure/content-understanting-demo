"""
Comprehensive demo script for Azure AI Content Understanding.
This script demonstrates the full capabilities of the service including
document analysis, audio processing, and custom analyzer creation.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

from content_understanding_client import create_client_from_env
from document_analysis_demo import DocumentAnalyzer
from audio_analysis_demo import AudioAnalyzer
from utils import (
    validate_azure_config,
    create_sample_analyzer_configs
)

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContentUnderstandingDemo:
    """
    Main demo class that orchestrates all Content Understanding capabilities.
    """
    
    def __init__(self):
        # Load environment
        load_dotenv()
        
        # Validate configuration
        self.config_validation = validate_azure_config()
        if not all([self.config_validation["endpoint"], 
                   self.config_validation["endpoint_format"]]):
            raise ValueError("Invalid Azure configuration. Please check your .env file.")
        
        # Create client and analyzers
        self.client = create_client_from_env()
        self.document_analyzer = DocumentAnalyzer(self.client)
        self.audio_analyzer = AudioAnalyzer(self.client)
        
        # Set up output directory
        self.output_dir = Path(os.getenv("DEFAULT_OUTPUT_DIR", "./output"))
        self.output_dir.mkdir(exist_ok=True)
        
        # Sample data URLs
        self.sample_data = {
            "invoice_pdf": "https://github.com/Azure-Samples/azure-ai-content-understanding-python/raw/refs/heads/main/data/invoice.pdf",
            "receipt_image": "https://github.com/Azure-Samples/azure-ai-content-understanding-python/raw/refs/heads/main/data/receipt.png",
            "audio_file": "https://github.com/Azure-Samples/azure-ai-content-understanding-python/raw/refs/heads/main/data/audio.wav",
        }
        
        print(f"Demo initialized. Output directory: {self.output_dir}")
    
    def run_configuration_check(self):
        """Run configuration validation and display results."""
        print("=== Configuration Check ===")
        
        checks = [
            ("Azure Endpoint", self.config_validation["endpoint"]),
            ("Endpoint Format", self.config_validation["endpoint_format"]),
            ("API Key/Auth", self.config_validation["api_key"] or True),  # Azure AD is alternative
            ("API Version", self.config_validation["api_version"]),
            ("Output Directory", self.config_validation["output_dir"])
        ]
        
        all_passed = True
        for check_name, passed in checks:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {check_name}: {status}")
            if not passed:
                all_passed = False
        
        if not all_passed:
            print("\n⚠️  Some configuration checks failed. Please review your .env file.")
            return False
        else:
            print("\n✅ All configuration checks passed!")
            return True
    
    def run_document_demos(self):
        """Run document analysis demonstrations."""
        print("\n=== Document Analysis Demos ===")
        
        demos = [
            {
                "name": "Prebuilt Document Analyzer",
                "method": self._demo_prebuilt_document,
                "description": "Analyze documents with general-purpose analyzer"
            },
            {
                "name": "Custom Invoice Analyzer", 
                "method": self._demo_custom_invoice_analyzer,
                "description": "Create and use a custom invoice analyzer"
            },
            {
                "name": "Document Structure Analysis",
                "method": self._demo_document_structure,
                "description": "Extract detailed document structure"
            }
        ]
        
        for demo in demos:
            print(f"\n--- {demo['name']} ---")
            print(f"Description: {demo['description']}")
            try:
                demo["method"]()
                print(f"✅ {demo['name']} completed successfully")
            except Exception as e:
                print(f"❌ {demo['name']} failed: {e}")
                logger.error(f"Demo {demo['name']} failed", exc_info=True)
    
    def run_audio_demos(self):
        """Run audio analysis demonstrations."""
        print("\n=== Audio Analysis Demos ===")
        
        demos = [
            {
                "name": "Prebuilt Audio Analyzer",
                "method": self._demo_prebuilt_audio,
                "description": "Basic audio transcription and analysis"
            },
            {
                "name": "Call Center Analyzer",
                "method": self._demo_call_center_analyzer,
                "description": "Specialized call center audio analysis"
            },
            {
                "name": "Custom Meeting Analyzer",
                "method": self._demo_custom_meeting_analyzer,
                "description": "Custom analyzer for meeting audio"
            }
        ]
        
        for demo in demos:
            print(f"\n--- {demo['name']} ---")
            print(f"Description: {demo['description']}")
            try:
                demo["method"]()
                print(f"✅ {demo['name']} completed successfully")
            except Exception as e:
                print(f"❌ {demo['name']} failed: {e}")
                logger.error(f"Demo {demo['name']} failed", exc_info=True)
    
    def run_analyzer_management_demo(self):
        """Demonstrate analyzer management operations."""
        print("\n=== Analyzer Management Demo ===")
        
        try:
            # List existing analyzers
            print("Listing available analyzers...")
            analyzers = self.client.list_analyzers()
            print(f"Found {len(analyzers)} analyzers")
            
            # Show sample analyzer configurations
            print("\nSample analyzer configurations:")
            configs = create_sample_analyzer_configs()
            for name, config in configs.items():
                print(f"  - {name}: {config['description']}")
            
            # Create a sample analyzer
            sample_analyzer_id = "demo-sample-analyzer"
            print(f"\nCreating sample analyzer: {sample_analyzer_id}")
            
            result = self.client.create_analyzer(
                sample_analyzer_id,
                configs["meeting_analyzer"]
            )
            
            if result.get("operation_location"):
                print("✅ Sample analyzer created successfully")
                
                # Clean up - delete the sample analyzer
                print("Cleaning up - deleting sample analyzer...")
                if self.client.delete_analyzer(sample_analyzer_id):
                    print("✅ Sample analyzer deleted successfully")
                else:
                    print("⚠️  Failed to delete sample analyzer")
            
        except Exception as e:
            print(f"❌ Analyzer management demo failed: {e}")
            logger.error("Analyzer management demo failed", exc_info=True)
    
    def _demo_prebuilt_document(self):
        """Demo prebuilt document analyzer."""
        result = self.document_analyzer.analyze_with_prebuilt_analyzer(
            document_url=self.sample_data["invoice_pdf"],
            output_file="demo_prebuilt_document.json"
        )
        
        # Display summary
        key_info = self.document_analyzer.extract_key_information(result)
        print(f"  Document analyzed: {key_info['analyzer_id']}")
        print(f"  Status: {key_info['status']}")
        if key_info['content_summary']['structure']:
            structure = key_info['content_summary']['structure']
            print(f"  Structure: {structure}")
        
        return result
    
    def _demo_custom_invoice_analyzer(self):
        """Demo custom invoice analyzer creation and usage."""
        analyzer_id = "demo-invoice-analyzer-main"
        
        # Create custom analyzer
        self.document_analyzer.create_custom_document_analyzer(
            analyzer_id=analyzer_id,
            description="Main demo invoice analyzer"
        )
        
        # Analyze document
        result = self.document_analyzer.analyze_with_custom_analyzer(
            analyzer_id=analyzer_id,
            document_url=self.sample_data["invoice_pdf"],
            output_file="demo_custom_invoice.json"
        )
        
        # Display extracted fields
        key_info = self.document_analyzer.extract_key_information(result)
        print(f"  Custom fields extracted:")
        for field_name, field_value in key_info['extracted_fields'].items():
            print(f"    {field_name}: {field_value}")
        
        return result
    
    def _demo_document_structure(self):
        """Demo detailed document structure analysis."""
        result = self.document_analyzer.analyze_with_prebuilt_analyzer(
            document_url=self.sample_data["receipt_image"],
            output_file="demo_document_structure.json"
        )
        
        # Extract detailed structure information
        contents = result.get("result", {}).get("contents", [])
        if contents:
            content = contents[0]
            
            structure_details = {
                "paragraphs": len(content.get("paragraphs", [])),
                "tables": len(content.get("tables", [])),
                "sections": len(content.get("sections", [])),
                "pages": len(content.get("pages", []))
            }
            
            print(f"  Document structure details:")
            for element, count in structure_details.items():
                print(f"    {element.capitalize()}: {count}")
        
        return result
    
    def _demo_prebuilt_audio(self):
        """Demo prebuilt audio analyzer."""
        result = self.audio_analyzer.analyze_with_prebuilt_analyzer(
            audio_url=self.sample_data["audio_file"],
            analyzer_type="general",
            output_file="demo_prebuilt_audio.json"
        )
        
        # Display audio insights
        insights = self.audio_analyzer.extract_audio_insights(result)
        print(f"  Audio duration: {insights['duration'].get('duration_minutes', 0):.1f} minutes")
        print(f"  Speakers detected: {len(insights['speakers'])}")
        
        # Show transcript preview
        if insights['transcript']:
            transcript_preview = self.audio_analyzer.generate_transcript_summary(
                insights['transcript'], 150
            )
            print(f"  Transcript preview: {transcript_preview}")
        
        return result
    
    def _demo_call_center_analyzer(self):
        """Demo call center audio analyzer."""
        result = self.audio_analyzer.analyze_with_prebuilt_analyzer(
            audio_url=self.sample_data["audio_file"],
            analyzer_type="call_center",
            output_file="demo_call_center.json"
        )
        
        # Display call center insights
        insights = self.audio_analyzer.extract_audio_insights(result)
        print(f"  Call analyzed with call center analyzer")
        print(f"  Duration: {insights['duration'].get('duration_minutes', 0):.1f} minutes")
        
        if insights['extracted_fields']:
            print(f"  Call center fields extracted:")
            for field_name, field_value in list(insights['extracted_fields'].items())[:3]:
                if isinstance(field_value, list):
                    value_str = ', '.join(field_value[:2])
                else:
                    value_str = str(field_value)[:50]
                print(f"    {field_name}: {value_str}...")
        
        return result
    
    def _demo_custom_meeting_analyzer(self):
        """Demo custom meeting audio analyzer."""
        analyzer_id = "demo-meeting-analyzer-main"
        
        # Create custom analyzer
        self.audio_analyzer.create_custom_audio_analyzer(
            analyzer_id=analyzer_id,
            base_analyzer="prebuilt-audioAnalyzer",
            description="Main demo meeting analyzer"
        )
        
        # Analyze audio
        result = self.audio_analyzer.analyze_with_custom_analyzer(
            analyzer_id=analyzer_id,
            audio_url=self.sample_data["audio_file"],
            output_file="demo_custom_meeting.json"
        )
        
        # Display custom analysis results
        insights = self.audio_analyzer.extract_audio_insights(result)
        print(f"  Meeting analysis completed")
        if insights['extracted_fields']:
            print(f"  Custom meeting fields:")
            for field_name, field_value in list(insights['extracted_fields'].items())[:3]:
                value_str = str(field_value)[:60] if not isinstance(field_value, list) else f"{len(field_value)} items"
                print(f"    {field_name}: {value_str}")
        
        return result
    
    def generate_demo_report(self):
        """Generate a summary report of all demo results."""
        print("\n=== Demo Report Generation ===")
        
        report = {
            "demo_timestamp": str(Path().cwd()),
            "configuration": self.config_validation,
            "output_directory": str(self.output_dir),
            "sample_data_urls": self.sample_data,
            "completed_demos": []
        }
        
        # Check which output files were created
        output_files = list(self.output_dir.glob("demo_*.json"))
        
        for output_file in output_files:
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                demo_summary = {
                    "file": output_file.name,
                    "status": result.get("status"),
                    "analyzer": result.get("result", {}).get("analyzerId"),
                    "created_at": result.get("result", {}).get("createdAt")
                }
                
                report["completed_demos"].append(demo_summary)
                
            except Exception as e:
                logger.warning(f"Could not process output file {output_file}: {e}")
        
        # Save report
        report_path = self.output_dir / "demo_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Demo report saved to: {report_path}")
        print(f"📄 Completed demos: {len(report['completed_demos'])}")
        print(f"📁 Output files: {len(output_files)}")
        
        return report


def main():
    """Main demo function."""
    print("🚀 Azure AI Content Understanding - Comprehensive Demo")
    print("=" * 60)
    
    try:
        # Initialize demo
        demo = ContentUnderstandingDemo()
        
        # Run configuration check
        if not demo.run_configuration_check():
            print("\n❌ Configuration check failed. Please fix your configuration and try again.")
            return
        
        # Run all demos
        demo.run_document_demos()
        demo.run_audio_demos()
        demo.run_analyzer_management_demo()
        
        # Generate report
        demo.generate_demo_report()
        
        print("\n🎉 All demos completed successfully!")
        print(f"📂 Check the output directory for results: {demo.output_dir}")
        
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        logger.error("Main demo failed", exc_info=True)
        
        print("\n🔧 Troubleshooting tips:")
        print("1. Check your .env file configuration")
        print("2. Verify your Azure AI Service endpoint and credentials")
        print("3. Ensure you have an active Azure subscription")
        print("4. Check network connectivity to Azure services")


if __name__ == "__main__":
    main()
