"""
Azure AI Content Understanding Client Library
A comprehensive Python client for Azure AI Content Understanding service.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Union, Any
import requests
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)


class ContentUnderstandingError(Exception):
    """Custom exception for Content Understanding operations"""
    pass


class AzureContentUnderstandingClient:
    """
    Azure AI Content Understanding client for analyzing documents, audio, images, and video.
    
    This client provides both synchronous and asynchronous methods for:
    - Creating and managing analyzers
    - Analyzing content with prebuilt or custom analyzers
    - Retrieving analysis results
    """
    
    def __init__(
        self,
        endpoint: str,
        credential: Optional[Union[str, AzureKeyCredential, DefaultAzureCredential]] = None,
        api_version: str = "2025-05-01-preview",
        timeout: int = 300,
        x_ms_useragent: Optional[str] = None
    ):
        """
        Initialize the Content Understanding client.
        
        Args:
            endpoint: Azure AI Service endpoint URL
            credential: Authentication credential (API key, Azure AD credential, or None for env vars)
            api_version: API version to use
            timeout: Request timeout in seconds
            x_ms_useragent: Custom user agent string
        """
        self.endpoint = endpoint.rstrip('/')
        self.api_version = api_version
        self.timeout = timeout
        
        # Set up authentication
        if isinstance(credential, str):
            self.credential = AzureKeyCredential(credential)
            self.auth_type = "key"
        elif isinstance(credential, AzureKeyCredential):
            self.credential = credential
            self.auth_type = "key"
        elif isinstance(credential, DefaultAzureCredential):
            self.credential = credential
            self.auth_type = "azure_ad"
        else:
            # Try to get from environment
            api_key = os.getenv("AZURE_AI_SERVICE_API_KEY")
            if api_key:
                self.credential = AzureKeyCredential(api_key)
                self.auth_type = "key"
            else:
                self.credential = DefaultAzureCredential()
                self.auth_type = "azure_ad"
        
        # Set up headers
        self.base_headers = {
            "Content-Type": "application/json"
        }
        
        if x_ms_useragent:
            self.base_headers["x-ms-useragent"] = x_ms_useragent
            
        # For Azure AD authentication
        if self.auth_type == "azure_ad":
            self.token_provider = get_bearer_token_provider(
                self.credential, 
                "https://cognitiveservices.azure.com/.default"
            )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with current authentication."""
        headers = self.base_headers.copy()
        
        if self.auth_type == "key":
            headers["Ocp-Apim-Subscription-Key"] = self.credential.key
        else:
            token = self.token_provider()
            headers["Authorization"] = f"Bearer {token}"
            
        return headers
    
    def _make_request(
        self, 
        method: str, 
        url: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> requests.Response:
        """Make HTTP request with proper error handling."""
        headers = self._get_headers()
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code >= 400:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise ContentUnderstandingError(error_msg)
                
            return response
            
        except requests.exceptions.Timeout:
            raise ContentUnderstandingError(f"Request timed out after {self.timeout} seconds")
        except requests.exceptions.RequestException as e:
            raise ContentUnderstandingError(f"Request failed: {str(e)}")
    
    def create_analyzer(
        self, 
        analyzer_id: str, 
        analyzer_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a custom analyzer.
        
        Args:
            analyzer_id: Unique identifier for the analyzer
            analyzer_config: Analyzer configuration including schema and settings
            
        Returns:
            Dictionary containing the operation status and location
        """
        url = f"{self.endpoint}/contentunderstanding/analyzers/{analyzer_id}"
        params = {"api-version": self.api_version}
        
        logger.info(f"Creating analyzer: {analyzer_id}")
        response = self._make_request("PUT", url, data=analyzer_config, params=params)
        
        result = {
            "status_code": response.status_code,
            "operation_location": response.headers.get("Operation-Location"),
            "request_id": response.headers.get("request-id")
        }
        
        if response.content:
            try:
                result.update(response.json())
            except json.JSONDecodeError:
                pass
                
        return result
    
    def get_analyzer_operation_status(self, operation_url: str) -> Dict[str, Any]:
        """
        Check the status of an analyzer creation operation.
        
        Args:
            operation_url: URL returned from create_analyzer operation
            
        Returns:
            Dictionary containing operation status
        """
        response = self._make_request("GET", operation_url)
        return response.json()
    
    def analyze_content(
        self, 
        analyzer_id: str, 
        content_url: Optional[str] = None,
        file_path: Optional[str] = None,
        additional_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze content using a specified analyzer.
        
        Args:
            analyzer_id: ID of the analyzer to use
            content_url: URL of content to analyze (mutually exclusive with file_path)
            file_path: Local file path to analyze (mutually exclusive with content_url)
            additional_params: Additional parameters for the analysis
            
        Returns:
            Dictionary containing analysis request ID and status
        """
        if not content_url and not file_path:
            raise ValueError("Either content_url or file_path must be provided")
        
        if content_url and file_path:
            raise ValueError("Only one of content_url or file_path can be provided")
        
        url = f"{self.endpoint}/contentunderstanding/analyzers/{analyzer_id}:analyze"
        params = {"api-version": self.api_version}
        
        if additional_params:
            params.update(additional_params)
        
        if content_url:
            data = {"url": content_url}
            logger.info(f"Analyzing content from URL with analyzer {analyzer_id}: {content_url}")
        else:
            # For file upload, we need to use multipart/form-data
            # This is a simplified version - for full file upload support,
            # you would need to implement multipart upload
            raise NotImplementedError("File upload not implemented in this demo. Use content_url instead.")
        
        response = self._make_request("POST", url, data=data, params=params)
        
        result = {
            "status_code": response.status_code,
            "request_id": response.headers.get("request-id"),
            "operation_location": response.headers.get("Operation-Location")
        }
        
        if response.content:
            try:
                result.update(response.json())
            except json.JSONDecodeError:
                pass
                
        return result
    
    def get_analysis_result(self, request_id: str) -> Dict[str, Any]:
        """
        Get the result of a content analysis operation.
        
        Args:
            request_id: Request ID returned from analyze_content
            
        Returns:
            Dictionary containing analysis results
        """
        url = f"{self.endpoint}/contentunderstanding/analyzerResults/{request_id}"
        params = {"api-version": self.api_version}
        
        response = self._make_request("GET", url, params=params)
        return response.json()
    
    def wait_for_analysis_completion(
        self, 
        request_id: str, 
        max_wait_time: int = 300,
        poll_interval: int = 5
    ) -> Dict[str, Any]:
        """
        Wait for analysis to complete and return the result.
        
        Args:
            request_id: Request ID from analyze_content
            max_wait_time: Maximum time to wait in seconds
            poll_interval: Time between status checks in seconds
            
        Returns:
            Dictionary containing final analysis results
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            result = self.get_analysis_result(request_id)
            status = result.get("status", "Unknown")
            
            if status.lower() == "succeeded":
                logger.info(f"Analysis completed successfully: {request_id}")
                return result
            elif status.lower() in ["failed", "cancelled"]:
                error_msg = f"Analysis failed with status: {status}"
                logger.error(error_msg)
                raise ContentUnderstandingError(error_msg)
            
            logger.info(f"Analysis in progress: {status}. Waiting {poll_interval} seconds...")
            time.sleep(poll_interval)
        
        raise ContentUnderstandingError(f"Analysis did not complete within {max_wait_time} seconds")
    
    def list_analyzers(self) -> List[Dict[str, Any]]:
        """
        List all available analyzers.
        
        Returns:
            List of analyzer definitions
        """
        url = f"{self.endpoint}/contentunderstanding/analyzers"
        params = {"api-version": self.api_version}
        
        response = self._make_request("GET", url, params=params)
        return response.json().get("value", [])
    
    def delete_analyzer(self, analyzer_id: str) -> bool:
        """
        Delete an analyzer.
        
        Args:
            analyzer_id: ID of the analyzer to delete
            
        Returns:
            True if deletion was successful
        """
        url = f"{self.endpoint}/contentunderstanding/analyzers/{analyzer_id}"
        params = {"api-version": self.api_version}
        
        try:
            response = self._make_request("DELETE", url, params=params)
            logger.info(f"Successfully deleted analyzer: {analyzer_id}")
            return True
        except ContentUnderstandingError as e:
            logger.error(f"Failed to delete analyzer {analyzer_id}: {e}")
            return False


# Prebuilt analyzer constants
PREBUILT_ANALYZERS = {
    "document": "prebuilt-documentAnalyzer",
    "image": "prebuilt-imageAnalyzer", 
    "audio": "prebuilt-audioAnalyzer",
    "video": "prebuilt-videoAnalyzer",
    "call_center": "prebuilt-callCenter"
}


def create_client_from_env() -> AzureContentUnderstandingClient:
    """
    Create a client using environment variables.
    
    Returns:
        Configured AzureContentUnderstandingClient instance
    """
    endpoint = os.getenv("AZURE_AI_SERVICE_ENDPOINT")
    if not endpoint:
        raise ValueError("AZURE_AI_SERVICE_ENDPOINT environment variable is required")
    
    api_version = os.getenv("AZURE_AI_SERVICE_API_VERSION", "2025-05-01-preview")
    
    return AzureContentUnderstandingClient(
        endpoint=endpoint,
        api_version=api_version,
        x_ms_useragent="azure-ai-content-understanding-demo/1.0"
    )
