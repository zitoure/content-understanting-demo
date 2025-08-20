"""
Batch Processing Demo for Azure AI Content Understanding

This script demonstrates how to process multiple files efficiently using
concurrent requests, progress tracking, and error handling.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from tqdm import tqdm

from content_understanding_client import AzureContentUnderstandingClient, create_client_from_env
from utils import get_file_type, save_results_to_file, validate_url

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Batch processor for Content Understanding operations.
    Handles multiple files with concurrent processing and error recovery.
    """
    
    def __init__(
        self,
        client: AzureContentUnderstandingClient,
        max_workers: int = 5,
        retry_attempts: int = 3
    ):
        self.client = client
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.output_dir = Path(os.getenv("DEFAULT_OUTPUT_DIR", "./output"))
        self.batch_output_dir = self.output_dir / "batch_results"
        self.batch_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None
        }
    
    def process_file_batch(
        self,
        file_configs: List[Dict[str, Any]],
        analyzer_id: str,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of files concurrently.
        
        Args:
            file_configs: List of file configurations with 'url' and 'output_file'
            analyzer_id: Analyzer to use for processing
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of processing results
        """
        self.stats["total_files"] = len(file_configs)
        self.stats["start_time"] = time.time()
        
        logger.info(f"Starting batch processing of {len(file_configs)} files")
        logger.info(f"Using analyzer: {analyzer_id}")
        logger.info(f"Max concurrent workers: {self.max_workers}")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_config = {
                executor.submit(
                    self._process_single_file_with_retry,
                    config,
                    analyzer_id
                ): config for config in file_configs
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(file_configs), desc="Processing files") as pbar:
                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result["status"] == "success":
                            self.stats["successful"] += 1
                        else:
                            self.stats["failed"] += 1
                            
                        # Update progress
                        pbar.set_postfix({
                            "Success": self.stats["successful"],
                            "Failed": self.stats["failed"]
                        })
                        
                        if progress_callback:
                            progress_callback(result)
                            
                    except Exception as e:
                        error_result = {
                            "file_url": config.get("url", "unknown"),
                            "output_file": config.get("output_file", "unknown"),
                            "status": "error",
                            "error": str(e),
                            "analyzer_id": analyzer_id
                        }
                        results.append(error_result)
                        self.stats["failed"] += 1
                        logger.error(f"Failed to process {config.get('url', 'unknown')}: {e}")
                    
                    pbar.update(1)
        
        self.stats["end_time"] = time.time()
        
        # Save batch summary
        self._save_batch_summary(results, analyzer_id)
        
        return results
    
    def _process_single_file_with_retry(
        self,
        config: Dict[str, Any],
        analyzer_id: str
    ) -> Dict[str, Any]:
        """
        Process a single file with retry logic.
        
        Args:
            config: File configuration
            analyzer_id: Analyzer to use
            
        Returns:
            Processing result
        """
        file_url = config["url"]
        output_file = config["output_file"]
        
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                logger.debug(f"Processing {file_url} (attempt {attempt + 1})")
                
                # Start analysis
                analysis_request = self.client.analyze_content(
                    analyzer_id=analyzer_id,
                    content_url=file_url
                )
                
                request_id = analysis_request["request_id"]
                
                # Wait for completion
                result = self.client.wait_for_analysis_completion(
                    request_id,
                    max_wait_time=600,  # 10 minutes
                    poll_interval=5
                )
                
                # Save result
                output_path = self.batch_output_dir / output_file
                save_results_to_file(result, output_path, "json")
                
                return {
                    "file_url": file_url,
                    "output_file": output_file,
                    "output_path": str(output_path),
                    "status": "success",
                    "request_id": request_id,
                    "analyzer_id": analyzer_id,
                    "result": result
                }
                
            except Exception as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed for {file_url}, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All attempts failed for {file_url}: {e}")
        
        return {
            "file_url": file_url,
            "output_file": output_file,
            "status": "failed",
            "error": str(last_error),
            "analyzer_id": analyzer_id
        }
    
    def _save_batch_summary(self, results: List[Dict[str, Any]], analyzer_id: str):
        """Save batch processing summary."""
        processing_time = self.stats["end_time"] - self.stats["start_time"]
        
        summary = {
            "analyzer_id": analyzer_id,
            "processing_time_seconds": processing_time,
            "statistics": self.stats.copy(),
            "success_rate": (self.stats["successful"] / self.stats["total_files"]) * 100,
            "results": results
        }
        
        summary_path = self.batch_output_dir / "batch_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch summary saved to: {summary_path}")
    
    def create_file_configs_from_urls(
        self,
        urls: List[str],
        base_filename: str = "result"
    ) -> List[Dict[str, Any]]:
        """
        Create file configurations from a list of URLs.
        
        Args:
            urls: List of file URLs
            base_filename: Base name for output files
            
        Returns:
            List of file configurations
        """
        configs = []
        
        for i, url in enumerate(urls):
            file_type = get_file_type(url)
            config = {
                "url": url,
                "output_file": f"{base_filename}_{i+1:03d}_{file_type}.json",
                "file_type": file_type,
                "index": i
            }
            configs.append(config)
        
        return configs
    
    def validate_batch_inputs(self, file_configs: List[Dict[str, Any]]) -> Tuple[List[Dict], List[str]]:
        """
        Validate batch input configurations.
        
        Args:
            file_configs: List of file configurations
            
        Returns:
            Tuple of (valid_configs, validation_errors)
        """
        valid_configs = []
        errors = []
        
        for i, config in enumerate(file_configs):
            config_errors = []
            
            # Check required fields
            if "url" not in config:
                config_errors.append("Missing 'url' field")
            elif not validate_url(config["url"]):
                config_errors.append(f"Invalid or inaccessible URL: {config['url']}")
            
            if "output_file" not in config:
                config_errors.append("Missing 'output_file' field")
            
            if config_errors:
                errors.append(f"Config {i}: {', '.join(config_errors)}")
            else:
                valid_configs.append(config)
        
        return valid_configs, errors


def create_sample_batch_config() -> Dict[str, Any]:
    """Create a sample batch processing configuration."""
    return {
        "analyzer_id": "prebuilt-documentAnalyzer",
        "files": [
            {
                "url": "https://github.com/Azure-Samples/azure-ai-content-understanding-python/raw/refs/heads/main/data/invoice.pdf",
                "output_file": "batch_invoice_001.json",
                "file_type": "document"
            },
            {
                "url": "https://github.com/Azure-Samples/azure-ai-content-understanding-python/raw/refs/heads/main/data/receipt.png",
                "output_file": "batch_receipt_002.json",
                "file_type": "image"
            }
        ],
        "settings": {
            "max_workers": 3,
            "retry_attempts": 2,
            "timeout_per_file": 300
        }
    }


def demo_document_batch_processing():
    """Demonstrate batch processing of document files."""
    print("=== Document Batch Processing Demo ===")
    
    # Sample document URLs
    document_urls = [
        "https://github.com/Azure-Samples/azure-ai-content-understanding-python/raw/refs/heads/main/data/invoice.pdf",
        "https://github.com/Azure-Samples/azure-ai-content-understanding-python/raw/refs/heads/main/data/receipt.png",
        # Add more URLs as needed
    ]
    
    try:
        client = create_client_from_env()
        processor = BatchProcessor(client, max_workers=3)
        
        # Create file configurations
        file_configs = processor.create_file_configs_from_urls(
            document_urls,
            "doc_batch"
        )
        
        print(f"Processing {len(file_configs)} documents...")
        
        # Validate inputs
        valid_configs, errors = processor.validate_batch_inputs(file_configs)
        if errors:
            print("Validation errors:")
            for error in errors:
                print(f"  - {error}")
            return
        
        # Process batch
        def progress_callback(result):
            status = "✅" if result["status"] == "success" else "❌"
            print(f"  {status} {result['file_url'].split('/')[-1]}")
        
        results = processor.process_file_batch(
            valid_configs,
            "prebuilt-documentAnalyzer",
            progress_callback
        )
        
        # Display summary
        print(f"\nBatch Processing Summary:")
        print(f"  Total files: {processor.stats['total_files']}")
        print(f"  Successful: {processor.stats['successful']}")
        print(f"  Failed: {processor.stats['failed']}")
        print(f"  Success rate: {(processor.stats['successful'] / processor.stats['total_files']) * 100:.1f}%")
        print(f"  Processing time: {processor.stats['end_time'] - processor.stats['start_time']:.1f} seconds")
        print(f"  Results saved to: {processor.batch_output_dir}")
        
    except Exception as e:
        print(f"Batch processing demo failed: {e}")


def demo_audio_batch_processing():
    """Demonstrate batch processing of audio files."""
    print("\n=== Audio Batch Processing Demo ===")
    
    # Sample audio URLs (you would replace with real audio files)
    audio_urls = [
        "https://github.com/Azure-Samples/azure-ai-content-understanding-python/raw/refs/heads/main/data/audio.wav",
        # Add more audio URLs as needed
    ]
    
    try:
        client = create_client_from_env()
        processor = BatchProcessor(client, max_workers=2)  # Fewer workers for audio (longer processing)
        
        # Create file configurations
        file_configs = processor.create_file_configs_from_urls(
            audio_urls,
            "audio_batch"
        )
        
        print(f"Processing {len(file_configs)} audio files...")
        
        # Process batch with call center analyzer
        results = processor.process_file_batch(
            file_configs,
            "prebuilt-callCenter"
        )
        
        # Display summary
        print(f"\nAudio Batch Processing Summary:")
        print(f"  Total files: {processor.stats['total_files']}")
        print(f"  Successful: {processor.stats['successful']}")
        print(f"  Failed: {processor.stats['failed']}")
        print(f"  Processing time: {processor.stats['end_time'] - processor.stats['start_time']:.1f} seconds")
        
    except Exception as e:
        print(f"Audio batch processing demo failed: {e}")


def main():
    """Main demo function for batch processing."""
    load_dotenv()
    
    print("🔄 Azure AI Content Understanding - Batch Processing Demo")
    print("=" * 60)
    
    try:
        # Run document batch demo
        demo_document_batch_processing()
        
        # Run audio batch demo
        demo_audio_batch_processing()
        
        print("\n✅ Batch processing demos completed!")
        print("📁 Check the ./output/batch_results directory for detailed results")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        logger.error("Batch processing demo failed", exc_info=True)


if __name__ == "__main__":
    main()
