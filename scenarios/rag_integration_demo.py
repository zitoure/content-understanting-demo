"""
RAG (Retrieval-Augmented Generation) Integration Demo for Azure AI Content Understanding

This script demonstrates how to integrate Content Understanding with RAG systems,
including Azure AI Search integration and vector embedding preparation.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

from content_understanding_client import AzureContentUnderstandingClient, create_client_from_env
from utils import extract_transcript_text

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGIntegrator:
    """
    RAG integration helper for Content Understanding.
    Prepares extracted content for vector databases and search systems.
    """
    
    def __init__(self, client: AzureContentUnderstandingClient):
        self.client = client
        self.output_dir = Path(os.getenv("DEFAULT_OUTPUT_DIR", "./output"))
        self.rag_output_dir = self.output_dir / "rag_data"
        self.rag_output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_content_for_rag(
        self,
        analysis_result: Dict[str, Any],
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Extract and chunk content for RAG systems.
        
        Args:
            analysis_result: Content Understanding analysis result
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of content chunks with metadata
        """
        chunks = []
        
        contents = analysis_result.get("result", {}).get("contents", [])
        if not contents:
            return chunks
        
        content = contents[0]
        analyzer_id = analysis_result.get("result", {}).get("analyzerId", "unknown")
        
        # Extract main content based on type
        if "markdown" in content:
            text_content = content["markdown"]
            content_type = "document"
        elif "transcriptPhrases" in content:
            text_content = extract_transcript_text(analysis_result)
            content_type = "audio"
        else:
            text_content = str(content)
            content_type = "unknown"
        
        # Create chunks
        if len(text_content) <= chunk_size:
            # Single chunk
            chunk = self._create_chunk(
                text_content,
                content,
                analyzer_id,
                content_type,
                0,
                len(text_content)
            )
            chunks.append(chunk)
        else:
            # Multiple chunks with overlap
            start = 0
            chunk_index = 0
            
            while start < len(text_content):
                end = min(start + chunk_size, len(text_content))
                
                # Find good break point (sentence or paragraph end)
                if end < len(text_content):
                    for break_char in ['\n\n', '. ', '.\n', '! ', '? ']:
                        break_pos = text_content.rfind(break_char, start, end)
                        if break_pos > start + chunk_size // 2:
                            end = break_pos + len(break_char)
                            break
                
                chunk_text = text_content[start:end].strip()
                if chunk_text:
                    chunk = self._create_chunk(
                        chunk_text,
                        content,
                        analyzer_id,
                        content_type,
                        start,
                        end,
                        chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Move to next chunk with overlap
                start = max(start + 1, end - overlap)
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        content: Dict[str, Any],
        analyzer_id: str,
        content_type: str,
        start_pos: int,
        end_pos: int,
        chunk_index: int = 0
    ) -> Dict[str, Any]:
        """Create a chunk object with metadata."""
        # Extract fields for metadata
        fields = content.get("fields", {})
        metadata = {}
        
        for field_name, field_data in fields.items():
            field_type = field_data.get("type")
            if field_type == "string":
                metadata[field_name] = field_data.get("valueString")
            elif field_type == "array":
                metadata[field_name] = field_data.get("valueArray", [])
            elif field_type in ["number", "integer"]:
                metadata[field_name] = field_data.get(f"value{field_type.capitalize()}")
            elif field_type == "boolean":
                metadata[field_name] = field_data.get("valueBoolean")
        
        chunk = {
            "id": f"{analyzer_id}_{chunk_index}_{start_pos}_{end_pos}",
            "text": text,
            "metadata": {
                "analyzer_id": analyzer_id,
                "content_type": content_type,
                "chunk_index": chunk_index,
                "start_position": start_pos,
                "end_position": end_pos,
                "chunk_length": len(text),
                **metadata
            }
        }
        
        # Add content-specific metadata
        if content_type == "document":
            chunk["metadata"]["page_count"] = len(content.get("pages", []))
            chunk["metadata"]["has_tables"] = len(content.get("tables", [])) > 0
            chunk["metadata"]["paragraph_count"] = len(content.get("paragraphs", []))
            
        elif content_type == "audio":
            chunk["metadata"]["duration_ms"] = content.get("endTimeMs", 0) - content.get("startTimeMs", 0)
            chunk["metadata"]["speaker_count"] = len(set(
                phrase.get("speaker", "") for phrase in content.get("transcriptPhrases", [])
            ))
        
        return chunk
    
    def prepare_for_azure_search(
        self,
        chunks: List[Dict[str, Any]],
        index_name: str = "content-understanding-index"
    ) -> Dict[str, Any]:
        """
        Prepare chunks for Azure AI Search indexing.
        
        Args:
            chunks: Content chunks from extract_content_for_rag
            index_name: Name of the search index
            
        Returns:
            Azure Search index configuration and documents
        """
        # Define index schema
        index_schema = {
            "name": index_name,
            "fields": [
                {"name": "id", "type": "Edm.String", "key": True, "searchable": False},
                {"name": "content", "type": "Edm.String", "searchable": True},
                {"name": "analyzer_id", "type": "Edm.String", "filterable": True},
                {"name": "content_type", "type": "Edm.String", "filterable": True},
                {"name": "chunk_index", "type": "Edm.Int32", "filterable": True},
                {"name": "chunk_length", "type": "Edm.Int32", "filterable": True},
                
                # Vector field for semantic search (if using vector search)
                {
                    "name": "contentVector",
                    "type": "Collection(Edm.Single)",
                    "searchable": True,
                    "vectorSearchDimensions": 1536,  # For text-embedding-3-small
                    "vectorSearchProfileName": "default-vector-profile"
                }
            ],
            "suggesters": [
                {
                    "name": "content-suggester",
                    "searchMode": "analyzingInfixMatching",
                    "sourceFields": ["content"]
                }
            ]
        }
        
        # Prepare documents for indexing
        documents = []
        for chunk in chunks:
            doc = {
                "id": chunk["id"],
                "content": chunk["text"],
                "analyzer_id": chunk["metadata"]["analyzer_id"],
                "content_type": chunk["metadata"]["content_type"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "chunk_length": chunk["metadata"]["chunk_length"]
            }
            
            # Add other metadata fields as dynamic fields
            for key, value in chunk["metadata"].items():
                if key not in ["analyzer_id", "content_type", "chunk_index", "chunk_length"]:
                    if isinstance(value, (str, int, float, bool)):
                        doc[f"metadata_{key}"] = value
            
            documents.append(doc)
        
        return {
            "index_schema": index_schema,
            "documents": documents,
            "document_count": len(documents),
            "total_text_length": sum(len(chunk["text"]) for chunk in chunks)
        }
    
    def create_knowledge_base_entry(
        self,
        analysis_result: Dict[str, Any],
        source_url: str,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a knowledge base entry from analysis results.
        
        Args:
            analysis_result: Content Understanding analysis result
            source_url: URL of the original content
            title: Optional title for the entry
            
        Returns:
            Knowledge base entry
        """
        contents = analysis_result.get("result", {}).get("contents", [])
        if not contents:
            return {}
        
        content = contents[0]
        analyzer_id = analysis_result.get("result", {}).get("analyzerId", "unknown")
        created_at = analysis_result.get("result", {}).get("createdAt")
        
        # Extract summary text
        summary = ""
        if "fields" in content:
            summary_field = content["fields"].get("Summary", {})
            summary = summary_field.get("valueString", "")
        
        if not summary and "markdown" in content:
            # Generate summary from first paragraph
            markdown = content["markdown"]
            lines = markdown.split('\n')
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    summary = line.strip()[:500] + "..."
                    break
        
        # Extract key fields
        extracted_fields = {}
        if "fields" in content:
            for field_name, field_data in content["fields"].items():
                field_type = field_data.get("type")
                if field_type == "string":
                    extracted_fields[field_name] = field_data.get("valueString")
                elif field_type == "array":
                    extracted_fields[field_name] = field_data.get("valueArray", [])
        
        # Create knowledge base entry
        kb_entry = {
            "id": f"kb_{analyzer_id}_{hash(source_url) % 100000}",
            "title": title or f"Content from {source_url.split('/')[-1]}",
            "summary": summary,
            "source_url": source_url,
            "analyzer_id": analyzer_id,
            "created_at": created_at,
            "content_type": self._determine_content_type(content),
            "extracted_fields": extracted_fields,
            "full_content": content.get("markdown", ""),
            "confidence_scores": self._extract_confidence_scores(content)
        }
        
        return kb_entry
    
    def _determine_content_type(self, content: Dict[str, Any]) -> str:
        """Determine content type from analysis result."""
        if "transcriptPhrases" in content:
            return "audio"
        elif "pages" in content:
            return "document"
        elif "width" in content and "height" in content:
            return "image"
        else:
            return "unknown"
    
    def _extract_confidence_scores(self, content: Dict[str, Any]) -> Dict[str, float]:
        """Extract confidence scores from content."""
        scores = {}
        
        if "fields" in content:
            for field_name, field_data in content["fields"].items():
                confidence = field_data.get("confidence")
                if confidence is not None:
                    scores[field_name] = confidence
        
        return scores


def demo_document_rag_integration():
    """Demonstrate RAG integration for documents."""
    print("=== Document RAG Integration Demo ===")
    
    try:
        client = create_client_from_env()
        rag_integrator = RAGIntegrator(client)
        
        # Analyze a document
        sample_url = "https://github.com/Azure-Samples/azure-ai-content-understanding-python/raw/refs/heads/main/data/invoice.pdf"
        
        print("1. Analyzing document...")
        analysis_request = client.analyze_content(
            analyzer_id="prebuilt-documentAnalyzer",
            content_url=sample_url
        )
        
        result = client.wait_for_analysis_completion(analysis_request["request_id"])
        
        # Extract content for RAG
        print("2. Extracting content for RAG...")
        chunks = rag_integrator.extract_content_for_rag(result, chunk_size=500)
        print(f"   Created {len(chunks)} content chunks")
        
        # Prepare for Azure Search
        print("3. Preparing for Azure AI Search...")
        search_config = rag_integrator.prepare_for_azure_search(chunks)
        print(f"   Prepared {search_config['document_count']} documents for indexing")
        
        # Create knowledge base entry
        print("4. Creating knowledge base entry...")
        kb_entry = rag_integrator.create_knowledge_base_entry(
            result,
            sample_url,
            "Sample Invoice Document"
        )
        
        # Save RAG data
        rag_data = {
            "source_url": sample_url,
            "analysis_result": result,
            "chunks": chunks,
            "search_config": search_config,
            "knowledge_base_entry": kb_entry
        }
        
        output_path = rag_integrator.rag_output_dir / "document_rag_data.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rag_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Document RAG integration completed")
        print(f"   Chunks created: {len(chunks)}")
        print(f"   Knowledge base entry: {kb_entry['title']}")
        print(f"   Data saved to: {output_path}")
        
        return rag_data
        
    except Exception as e:
        print(f"Document RAG integration demo failed: {e}")
        return None


def demo_audio_rag_integration():
    """Demonstrate RAG integration for audio."""
    print("\n=== Audio RAG Integration Demo ===")
    
    try:
        client = create_client_from_env()
        rag_integrator = RAGIntegrator(client)
        
        # Analyze audio
        sample_url = "https://github.com/Azure-Samples/azure-ai-content-understanding-python/raw/refs/heads/main/data/audio.wav"
        
        print("1. Analyzing audio...")
        analysis_request = client.analyze_content(
            analyzer_id="prebuilt-callCenter",
            content_url=sample_url
        )
        
        result = client.wait_for_analysis_completion(
            analysis_request["request_id"],
            max_wait_time=600
        )
        
        # Extract content for RAG
        print("2. Extracting audio content for RAG...")
        chunks = rag_integrator.extract_content_for_rag(result, chunk_size=1000)
        print(f"   Created {len(chunks)} content chunks")
        
        # Create knowledge base entry
        print("3. Creating audio knowledge base entry...")
        kb_entry = rag_integrator.create_knowledge_base_entry(
            result,
            sample_url,
            "Sample Call Recording"
        )
        
        # Save RAG data
        rag_data = {
            "source_url": sample_url,
            "analysis_result": result,
            "chunks": chunks,
            "knowledge_base_entry": kb_entry
        }
        
        output_path = rag_integrator.rag_output_dir / "audio_rag_data.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rag_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Audio RAG integration completed")
        print(f"   Transcript chunks: {len(chunks)}")
        print(f"   Knowledge base entry: {kb_entry['title']}")
        print(f"   Data saved to: {output_path}")
        
        return rag_data
        
    except Exception as e:
        print(f"Audio RAG integration demo failed: {e}")
        return None


def generate_rag_integration_guide():
    """Generate a guide for RAG integration."""
    guide = {
        "rag_integration_guide": {
            "overview": "Guide for integrating Azure AI Content Understanding with RAG systems",
            "steps": [
                {
                    "step": 1,
                    "title": "Content Analysis",
                    "description": "Use Content Understanding to analyze documents, audio, or other content",
                    "code_example": "result = client.analyze_content(analyzer_id, content_url)"
                },
                {
                    "step": 2,
                    "title": "Content Extraction",
                    "description": "Extract and chunk content for vector databases",
                    "code_example": "chunks = rag_integrator.extract_content_for_rag(result)"
                },
                {
                    "step": 3,
                    "title": "Vector Embedding",
                    "description": "Generate embeddings for semantic search (use Azure OpenAI)",
                    "code_example": "embeddings = openai_client.embeddings.create(input=chunk_text)"
                },
                {
                    "step": 4,
                    "title": "Index Creation",
                    "description": "Create search index in Azure AI Search or other vector DB",
                    "code_example": "search_client.create_index(index_schema)"
                },
                {
                    "step": 5,
                    "title": "Document Ingestion",
                    "description": "Upload documents with embeddings to search index",
                    "code_example": "search_client.upload_documents(documents)"
                }
            ],
            "best_practices": [
                "Use appropriate chunk sizes (500-1000 chars for documents, longer for audio)",
                "Include overlap between chunks for better context preservation",
                "Add rich metadata from extracted fields for better filtering",
                "Use confidence scores to prioritize high-quality extractions",
                "Implement proper error handling and retry logic"
            ]
        }
    }
    
    return guide


def main():
    """Main demo function for RAG integration."""
    load_dotenv()
    
    print("🔗 Azure AI Content Understanding - RAG Integration Demo")
    print("=" * 60)
    
    try:
        # Run document RAG demo
        document_rag_data = demo_document_rag_integration()
        
        # Run audio RAG demo
        audio_rag_data = demo_audio_rag_integration()
        
        # Generate integration guide
        print("\n=== RAG Integration Guide ===")
        guide = generate_rag_integration_guide()
        
        output_dir = Path(os.getenv("DEFAULT_OUTPUT_DIR", "./output"))
        rag_output_dir = output_dir / "rag_data"
        
        guide_path = rag_output_dir / "rag_integration_guide.json"
        with open(guide_path, 'w', encoding='utf-8') as f:
            json.dump(guide, f, indent=2, ensure_ascii=False)
        
        print(f"📘 RAG integration guide saved to: {guide_path}")
        print("\n✅ RAG integration demos completed!")
        print(f"📁 Check the {rag_output_dir} directory for all RAG data")
        
    except Exception as e:
        print(f"RAG integration demo failed: {e}")
        logger.error("RAG integration demo failed", exc_info=True)


if __name__ == "__main__":
    main()
