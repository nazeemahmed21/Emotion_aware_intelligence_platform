#!/usr/bin/env python3
"""
Hume AI Client Module

Provides direct communication with Hume AI API for emotion analysis.
Supports local files, S3 URLs, and various granularity levels.
All settings configurable through audio_analysis.json.
"""

import os
import logging
import asyncio
import json
import base64
import boto3
from enum import Enum
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
import aiohttp
from datetime import datetime, timedelta
from pathlib import Path
from botocore.config import Config

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

class GranularityLevel(Enum):
    """Supported granularity levels for Hume AI analysis."""
    WORD = "word"
    SENTENCE = "sentence"
    UTTERANCE = "utterance"
    TURN = "conversational_turn"  # Maps to 'conversational_turn' in API

@dataclass
class HumeConfig:
    """Configuration for Hume AI client."""
    api_key: str
    secret_key: Optional[str] = None
    webhook_url: Optional[str] = None
    max_retries: int = 3
    retry_delay: int = 5
    job_check_interval: int = 30
    job_timeout: int = 7200  # 2 hours

    @classmethod
    def from_config_file(cls, config_path: str = "config/analysis_config/audio_analysis.json") -> "HumeConfig":
        """
        Load configuration with environment variables taking priority over JSON config.
        
        Priority order:
        1. Environment variables (HUME_API_KEY, HUME_SECRET_KEY, HUME_WEBHOOK_URL)
        2. JSON configuration file
        3. Defaults
        """
        try:
            # Always try to load JSON config for non-credential settings
            hume_config = {}
            config_file = Path(config_path)
            if not config_file.exists():
                # Try relative to script location
                script_dir = Path(__file__).parent.parent.parent
                config_file = script_dir / "config" / "analysis_config" / "audio_analysis.json"
            
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                hume_config = config.get('hume_ai', {})
                LOGGER.info(f"Loaded Hume config from: {config_file}")
            else:
                LOGGER.warning(f"Config file not found: {config_path}, using environment variables and defaults")
            
            # Prioritize environment variables for credentials
            api_key = os.getenv('HUME_API_KEY') or hume_config.get('api_key', '')
            secret_key = os.getenv('HUME_SECRET_KEY') or hume_config.get('secret_key')
            webhook_url = os.getenv('HUME_WEBHOOK_URL') or hume_config.get('webhook_url')
            
            # Validate that we have required credentials
            if not api_key:
                raise ValueError("HUME_API_KEY must be set in environment variables or config file")
            
            LOGGER.info(f"🔑 Using API key from: {'environment variable' if os.getenv('HUME_API_KEY') else 'config file'}")
            if secret_key:
                LOGGER.info(f"🔑 Using secret key from: {'environment variable' if os.getenv('HUME_SECRET_KEY') else 'config file'}")
            
            return cls(
                api_key=api_key,
                secret_key=secret_key,
                webhook_url=webhook_url,
                max_retries=hume_config.get('max_retries', 3),
                retry_delay=hume_config.get('retry_delay_seconds', 5),
                job_check_interval=hume_config.get('job_check_interval_seconds', 30),
                job_timeout=hume_config.get('job_timeout_seconds', 7200)
            )
            
        except Exception as e:
            LOGGER.error(f"Error loading config: {e}")
            # Final fallback to environment variables only
            api_key = os.getenv('HUME_API_KEY', '')
            if not api_key:
                raise ValueError("HUME_API_KEY must be set in environment variables when config file is unavailable")
            
            return cls(
                api_key=api_key,
                secret_key=os.getenv('HUME_SECRET_KEY'),
                webhook_url=os.getenv('HUME_WEBHOOK_URL')
            )

def get_s3_presigned_url(bucket: str, key: str, expiration: int = 3600) -> str:
    """
    Generate a presigned URL for accessing an S3 object.
    Ensures an HTTPS link with virtual-hosted style.
    """
    s3_client = boto3.client(
        's3',
        config=Config(
            signature_version='s3v4',
            s3={'addressing_style': 'virtual'}
        )
    )
    
    url = s3_client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': bucket,
            'Key': key,
            'ResponseContentType': 'audio/mpeg'
        },
        ExpiresIn=expiration
    )
    
    if not url.startswith('https://'):
        raise ValueError(f"Generated URL is not HTTPS: {url}")
    
    LOGGER.debug(f"Generated presigned URL: {url}")
    return url

class HumeClient:
    """
    A client wrapper to interact with the Hume AI Batch Inference endpoints.
    Supports uploading local or S3-based files, monitoring job status,
    and downloading predictions/artifacts.
    """

    def __init__(self, config: HumeConfig):
        self.config = config
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        
    async def _get_access_token(self) -> Optional[str]:
        """
        Obtain an OAuth2 access token if a secret_key is configured.
        For most server-side usage, an API key alone is sufficient.
        """
        if not self.config.secret_key:
            return None

        # Reuse token if not expired
        if (self._access_token and self._token_expiry
                and datetime.now() < self._token_expiry - timedelta(minutes=5)):
            return self._access_token

        try:
            auth_string = f"{self.config.api_key}:{self.config.secret_key}"
            basic_auth = base64.b64encode(auth_string.encode()).decode()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.hume.ai/oauth2-cc/token",
                    headers={"Authorization": f"Basic {basic_auth}"},
                    data={"grant_type": "client_credentials"}
                ) as response:
                    response.raise_for_status()
                    
                    data = await response.json()
                    self._access_token = data["access_token"]
            self._token_expiry = datetime.now() + timedelta(minutes=30)
            return self._access_token

        except Exception as e:
            LOGGER.error(f"Failed to get access token: {e}", exc_info=True)
            raise

    def _get_headers(self, content_type: Optional[str] = None) -> Dict[str, str]:
        """
        Returns a headers dict with the required API key and optional content type.
        """
        headers = {
            "X-Hume-Api-Key": self.config.api_key
        }
        if content_type:
            headers["Content-Type"] = content_type
        else:
            headers["Accept"] = "application/json"
        return headers

    async def submit_urls(self, urls: List[str], granularity: Optional[GranularityLevel] = None) -> str:
        """
        Submit a job to Hume using one or more HTTPS URLs.
        Optionally configure prosody/language granularity and speaker identification.
        """
        try:
            endpoint = "https://api.hume.ai/v0/batch/jobs"
            headers = self._get_headers("application/json")
            
            # Configure models according to API specification
            models_config = {
                "burst": {},
                "prosody": {
                    "granularity": granularity.value if granularity else "utterance",
                    "identify_speakers": False
                },
                "language": {
                    "granularity": granularity.value if granularity else "utterance",
                    "identify_speakers": False
                }
            }
            
            # Enable speaker identification for conversational turn
            if granularity and granularity.value == "conversational_turn":
                models_config["prosody"]["identify_speakers"] = True
                models_config["language"]["identify_speakers"] = True
            
            # Configure transcription if needed
            transcription_config = None
            if granularity and granularity.value == "conversational_turn":
                transcription_config = {
                    "identify_speakers": True,
                    "confidence_threshold": 0.3
                }
            
            # Validate URLs are HTTPS
            for u in urls:
                if not u.startswith('https://'):
                    raise ValueError(f"All URLs must be HTTPS. Invalid: {u}")
            
            payload = {
                "models": models_config,
                "urls": urls,
                "notify": False
            }
            
            # Add transcription config if needed
            if transcription_config:
                payload["transcription"] = transcription_config
            
            # Add webhook if configured
            if self.config.webhook_url and self.config.webhook_url.startswith(('http://', 'https://')):
                payload["notify"] = True
                payload["callback_url"] = self.config.webhook_url
            
            LOGGER.debug("Submitting payload to %s:\n%s", endpoint, json.dumps(payload, indent=2))
            
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=payload) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        LOGGER.error(f"API Error: {response.status} - {response_text}")
                        LOGGER.error(f"Request payload: {json.dumps(payload, indent=2)}")
                    response.raise_for_status()
                    
                    data = await response.json()
                    job_id = data["job_id"]
            LOGGER.info(f"Successfully created job: {job_id}")
            return job_id

        except Exception as e:
            LOGGER.error(f"Failed to submit URLs: {e}", exc_info=True)
            raise

    async def submit_s3_file(self, bucket: str, key: str, granularity: Optional[GranularityLevel] = None) -> str:
        """
        Downloads an S3 file locally, then submits it to Hume as a local file.
        Cleans up the local temp file afterwards.
        """
        try:
            temp_dir = Path("/tmp/hume_uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file = temp_dir / Path(key).name
            
            LOGGER.info(f"Downloading {bucket}/{key} to {temp_file}")
            s3_client = boto3.client('s3')
            s3_client.download_file(bucket, key, str(temp_file))
            
            try:
                return await self.submit_files([str(temp_file)], granularity)
            finally:
                if temp_file.exists():
                    temp_file.unlink()
                
        except Exception as e:
            LOGGER.error(f"Failed to submit S3 file {bucket}/{key}: {e}", exc_info=True)
            raise

    async def submit_files(self, files: List[str], granularity: Optional[GranularityLevel] = None) -> str:
        """
        Submit one or more local files via multipart/form-data.
        Optionally configures prosody & language granularity + speaker ID.
        """
        try:
            endpoint = "https://api.hume.ai/v0/batch/jobs"
            headers = self._get_headers()
            
            # Configure models according to API specification
            models_config = {
                "burst": {},
                "prosody": {
                    "granularity": granularity.value if granularity else "utterance",
                    "identify_speakers": False
                },
                "language": {
                    "granularity": granularity.value if granularity else "utterance", 
                    "identify_speakers": False
                }
            }
            
            # Enable speaker identification for conversational turn
            if granularity and granularity.value == "conversational_turn":
                models_config["prosody"]["identify_speakers"] = True
                models_config["language"]["identify_speakers"] = True
            
            transcription_config = None
            if granularity and granularity.value == "conversational_turn":
                transcription_config = {
                    "identify_speakers": True,
                    "confidence_threshold": 0.3
                }
            
            # Build request payload
            request_data = {
                "models": models_config
            }
            if transcription_config:
                request_data["transcription"] = transcription_config
            
            # Prepare multipart form data for aiohttp
            LOGGER.debug("Submitting request to %s:\n%s", endpoint, json.dumps(request_data, indent=2))
            
            async with aiohttp.ClientSession() as session:
                # Create multipart form data
                data = aiohttp.FormData()
                data.add_field('json', json.dumps(request_data))
                
                # Add files to form data
                for file_path in files:
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                    data.add_field('file', file_content, 
                                   filename=os.path.basename(file_path),
                                   content_type='audio/wav')
                
                async with session.post(endpoint, headers=headers, data=data) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        LOGGER.error(f"API Error: {response.status} - {response_text}")
                        LOGGER.error(f"Request data: {json.dumps(request_data, indent=2)}")
                    response.raise_for_status()
                    
                    result = await response.json()
                    job_id = result["job_id"]
            LOGGER.info(f"Successfully created job: {job_id}")
            return job_id

        except Exception as e:
            LOGGER.error(f"Failed to submit files: {e}", exc_info=True)
            raise

    async def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Fetches the current job status from the Hume API.
        """
        try:
            endpoint = f"https://api.hume.ai/v0/batch/jobs/{job_id}"
            headers = self._get_headers()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, headers=headers) as response:
                    response.raise_for_status()
                    return await response.json()

        except Exception as e:
            LOGGER.error(f"Failed to check job status: {e}", exc_info=True)
            raise

    async def wait_for_job(self, job_id: str) -> bool:
        """
        Polls job status until completion or timeout. Returns True if completed, False if failed.
        """
        start_time = datetime.now()
        retries = 0
        
        while True:
            if (datetime.now() - start_time).total_seconds() > self.config.job_timeout:
                raise TimeoutError(f"Job {job_id} exceeded timeout of {self.config.job_timeout} seconds")

            try:
                status_response = await self.check_job_status(job_id)
                job_status = status_response.get("state", {}).get("status")
                
                LOGGER.info(f"Job {job_id} status: {job_status}")
                
                if job_status == "COMPLETED":
                    LOGGER.info("Job completed, proceeding to download predictions...")
                    return True
                elif job_status == "FAILED":
                    error_msg = status_response.get("state", {}).get("error", "Unknown error")
                    LOGGER.error(f"Job {job_id} failed: {error_msg}")
                    return False
                elif job_status in ["QUEUED", "IN_PROGRESS"]:
                    LOGGER.info(f"Job {job_id} is {job_status}, waiting {self.config.job_check_interval} seconds...")
                    await asyncio.sleep(self.config.job_check_interval)
                else:
                    LOGGER.warning(f"Unknown job status: {job_status}. Retrying in {self.config.job_check_interval} seconds...")
                    await asyncio.sleep(self.config.job_check_interval)
                
            except Exception as e:
                retries += 1
                if retries >= self.config.max_retries:
                    raise
                LOGGER.warning(f"Retry {retries}/{self.config.max_retries} after error: {e}")
                await asyncio.sleep(self.config.retry_delay)

    async def get_job_predictions(self, job_id: str, format: str = "json") -> Union[Dict[str, Any], bytes]:
        """
        Downloads job predictions in either JSON or 'artifacts' (zip) format.
        """
        try:
            LOGGER.info(f"Downloading predictions for job {job_id}...")
            
            if format == "json":
                endpoint = f"https://api.hume.ai/v0/batch/jobs/{job_id}/predictions"
                headers = self._get_headers()
            else:  # "artifacts" or CSV format
                endpoint = f"https://api.hume.ai/v0/batch/jobs/{job_id}/artifacts"
                headers = {
                    "X-Hume-Api-Key": self.config.api_key,
                    "Accept": "application/octet-stream"
                }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, headers=headers) as response:
                    response.raise_for_status()
                    
                    LOGGER.info("Successfully downloaded predictions")
                    if format == "json":
                        return await response.json()
                    else:
                        return await response.read()

        except Exception as e:
            LOGGER.error(f"Failed to get job predictions: {e}", exc_info=True)
            raise

    def analyze_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the 'predictions' JSON into a simpler emotion summary.
        """
        try:
            LOGGER.debug(f"Analyzing predictions: {json.dumps(predictions, indent=2)}")
            
            predictions_list = predictions.get("results", {}).get("predictions", [])
            if not predictions_list:
                raise ValueError("No predictions found in response")
                
            # Initialize results dictionary
            results = {
                'prosody': [],
                'bursts': [],
                'language': []
            }
            
            # Parse each file's predictions
            for prediction in predictions_list:
                models = prediction.get("models", {})
                
                # Extract prosody predictions
                if "prosody" in models:
                    for group in models["prosody"].get("grouped_predictions", []):
                        for pred in group.get("predictions", []):
                            if "emotions" in pred:
                                results['prosody'].extend(pred["emotions"])
                
                # Extract burst predictions
                if "burst" in models:
                    for group in models["burst"].get("grouped_predictions", []):
                        for pred in group.get("predictions", []):
                            if "emotions" in pred:
                                results['bursts'].extend(pred["emotions"])
                
                # Extract language predictions
                if "language" in models:
                    for group in models["language"].get("grouped_predictions", []):
                        for pred in group.get("predictions", []):
                            if "emotions" in pred:
                                results['language'].extend(pred["emotions"])
            
            # Combine emotions across modalities
            combined_emotions = {}
            for modality_data in results.values():
                for emotion in modality_data:
                    name = emotion['name']
                    score = emotion['score']
                    if name not in combined_emotions:
                        combined_emotions[name] = []
                    combined_emotions[name].append(score)
            
            return {
                'detailed_results': results,
                'emotion_summary': {
                    emotion: sum(scores) / len(scores)
                    for emotion, scores in combined_emotions.items()
                    if scores
                }
            }

        except Exception as e:
            LOGGER.error(f"Failed to analyze predictions: {e}", exc_info=True)
            raise
