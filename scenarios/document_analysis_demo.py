"""
Document Analysis Demo for Azure AI Content Understanding

This script demonstrates how to analyze documents using both prebuilt and custom analyzers.
It supports various document formats including PDF, images, and Office documents.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dotenv import load_dotenv

from content_understanding_client import AzureContentUnderstandingClient, create_client_from_env

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentAnalyzer:
    """
    Document analyzer for Azure AI Content Understanding.
    Handles document processing with both prebuilt and custom analyzers.
    """
    
    def __init__(self, client: AzureContentUnderstandingClient):
        self.client = client
        self.output_dir = Path(os.getenv("DEFAULT_OUTPUT_DIR", "./output"))
        self.output_dir.mkdir(exist_ok=True)
    
    def analyze_with_prebuilt_analyzer(
        self, 
        document_url: str,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a document using the prebuilt document analyzer.
        
        Args:
            document_url: URL of the document to analyze
            output_file: Optional output file path for results
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Starting document analysis with prebuilt analyzer")
        logger.info(f"Document URL: {document_url}")
        
        # Start analysis
        analysis_request = self.client.analyze_content(
            analyzer_id="prebuilt-documentAnalyzer",
            content_url=document_url
        )
        
        request_id = analysis_request["request_id"]
        logger.info(f"Analysis started with request ID: {request_id}")
        
        # Wait for completion
        result = self.client.wait_for_analysis_completion(request_id)
        
        # Save results if output file specified
        if output_file:
            output_path = self.output_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_path}")
        
        return result
    
    def create_custom_document_analyzer(
        self, 
        analyzer_id: str,
        description: str = "Custom document analyzer",
        custom_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a custom document analyzer with specified fields.
        
        Args:
            analyzer_id: Unique identifier for the analyzer
            description: Description of the analyzer
            custom_fields: Custom field schema for extraction
            
        Returns:
            Analyzer creation result
        """
        # Default custom fields for invoice processing
        if custom_fields is None:
            custom_fields = {
                "VendorName": {
                    "type": "string",
                    "description": "Name of the vendor or company issuing the document"
                },
                "TotalAmount": {
                    "type": "number",
                    "description": "Total amount or cost from the document"
                },
                "DocumentDate": {
                    "type": "date",
                    "description": "Date of the document"
                },
                "DocumentType": {
                    "type": "string",
                    "enum": ["invoice", "receipt", "contract", "report", "other"],
                    "description": "Type of document"
                },
                "Summary": {
                    "type": "string",
                    "method": "generate",
                    "description": "Brief summary of the document content"
                },
                "KeyTerms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Important terms or keywords found in the document"
                }
            }
        
        analyzer_config = {
            "description": description,
            "baseAnalyzerId": "prebuilt-documentAnalyzer",
            "config": {
                "returnDetails": True,
                "enableFormula": False,
                "disableContentFiltering": False,
                "estimateFieldSourceAndConfidence": True,
                "tableFormat": "html"
            },
            "fieldSchema": {
                "fields": custom_fields
            }
        }
        
        logger.info(f"Creating custom analyzer: {analyzer_id}")
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
        document_url: str,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a document using a custom analyzer.
        
        Args:
            analyzer_id: ID of the custom analyzer to use
            document_url: URL of the document to analyze
            output_file: Optional output file path for results
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Analyzing document with custom analyzer: {analyzer_id}")
        logger.info(f"Document URL: {document_url}")
        
        # Start analysis
        analysis_request = self.client.analyze_content(
            analyzer_id=analyzer_id,
            content_url=document_url
        )
        
        request_id = analysis_request["request_id"]
        logger.info(f"Analysis started with request ID: {request_id}")
        
        # Wait for completion
        result = self.client.wait_for_analysis_completion(request_id)
        
        # Save results if output file specified
        if output_file:
            output_path = self.output_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_path}")
        
        return result
    
    def extract_key_information(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key information from analysis results for easy consumption.
        
        Args:
            result: Raw analysis results
            
        Returns:
            Simplified key information dictionary
        """
        extracted_info = {
            "status": result.get("status"),
            "analyzer_id": result.get("result", {}).get("analyzerId"),
            "created_at": result.get("result", {}).get("createdAt"),
            "content_summary": {},
            "extracted_fields": {},
            "warnings": result.get("result", {}).get("warnings", [])
        }
        
        contents = result.get("result", {}).get("contents", [])
        if contents:
            content = contents[0]  # Get first content item
            
            # Extract markdown content
            extracted_info["content_summary"]["markdown"] = content.get("markdown", "")[:500] + "..."
            
            # Extract custom fields
            fields = content.get("fields", {})
            for field_name, field_data in fields.items():
                if field_data.get("type") == "string":
                    extracted_info["extracted_fields"][field_name] = field_data.get("valueString")
                elif field_data.get("type") == "number":
                    extracted_info["extracted_fields"][field_name] = field_data.get("valueNumber")
                elif field_data.get("type") == "date":
                    extracted_info["extracted_fields"][field_name] = field_data.get("valueDate")
                elif field_data.get("type") == "array":
                    extracted_info["extracted_fields"][field_name] = field_data.get("valueArray", [])
            
            # Extract structure information
            structure_info = {}
            if "paragraphs" in content:
                structure_info["paragraph_count"] = len(content["paragraphs"])
            if "tables" in content:
                structure_info["table_count"] = len(content["tables"])
            if "sections" in content:
                structure_info["section_count"] = len(content["sections"])
            
            extracted_info["content_summary"]["structure"] = structure_info
        
        return extracted_info


def main():
    """Main demo function for document analysis."""
    # Load environment variables
    load_dotenv()
    
    # Create client
    try:
        client = create_client_from_env()
        analyzer = DocumentAnalyzer(client)
        
        print("=== Azure AI Content Understanding - Document Analysis Demo ===\n")
        
        # Sample document URLs (you can replace these with your own)
        sample_documents = {
            "invoice": "https://github.com/Azure-Samples/azure-ai-content-understanding-python/raw/refs/heads/main/data/invoice.pdf",
            "receipt": "https://github.com/Azure-Samples/azure-ai-content-understanding-python/raw/refs/heads/main/data/receipt.png"
        }
        
        # Demo 1: Analyze with prebuilt analyzer
        print("1. Analyzing document with prebuilt analyzer...")
        try:
            result = analyzer.analyze_with_prebuilt_analyzer(
                document_url=sample_documents["invoice"],
                output_file="prebuilt_analysis_result.json"
            )
            
            # Extract and display key information
            key_info = analyzer.extract_key_information(result)
            print(f"   Status: {key_info['status']}")
            print(f"   Analyzer: {key_info['analyzer_id']}")
            print(f"   Content preview: {key_info['content_summary']['markdown'][:200]}...")
            print(f"   Structure: {key_info['content_summary']['structure']}")
            print("   ✓ Prebuilt analysis completed successfully\n")
            
        except Exception as e:
            print(f"   ✗ Prebuilt analysis failed: {e}\n")
        
        # Demo 2: Create and use custom analyzer
        print("2. Creating custom document analyzer...")
        custom_analyzer_id = "demo-invoice-analyzer"
        
        try:
            # Create custom analyzer
            creation_result = analyzer.create_custom_document_analyzer(
                analyzer_id=custom_analyzer_id,
                description="Demo invoice analyzer with custom fields"
            )
            print(f"   ✓ Custom analyzer '{custom_analyzer_id}' created successfully\n")
            
            # Analyze with custom analyzer
            print("3. Analyzing document with custom analyzer...")
            custom_result = analyzer.analyze_with_custom_analyzer(
                analyzer_id=custom_analyzer_id,
                document_url=sample_documents["invoice"],
                output_file="custom_analysis_result.json"
            )
            
            # Extract and display custom field results
            key_info = analyzer.extract_key_information(custom_result)
            print(f"   Status: {key_info['status']}")
            print("   Extracted custom fields:")
            for field_name, field_value in key_info['extracted_fields'].items():
                print(f"     {field_name}: {field_value}")
            print("   ✓ Custom analysis completed successfully\n")
            
        except Exception as e:
            print(f"   ✗ Custom analyzer demo failed: {e}\n")
        
        print("=== Document Analysis Demo Completed ===")
        print(f"Results saved to: {analyzer.output_dir}")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Please check your configuration in .env file")


if __name__ == "__main__":
    main()
