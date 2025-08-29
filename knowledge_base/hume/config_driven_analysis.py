#!/usr/bin/env python3
"""
Config-Driven Hume AI Analysis

Processes audio segments based on audio_analysis.json configuration, combining segments
and providing analysis output matching iteration_1 format.
No CLI interface - all configuration through audio_analysis.json.
"""

import json
import logging
import sys
import asyncio
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import centralized selection config
from utils.selection_config import get_selected_sales_reps, get_sales_rep_display_names

from hume.hume_client import HumeClient, HumeConfig, GranularityLevel
from hume.core.config_loader import load_config
from hume.core.analysis_config import get_analysis_config
from hume.core.emotion_processor import (
    extract_emotions_from_predictions,
    extract_descriptions_from_predictions,
    create_emotions_dataframe,
    create_descriptions_dataframe,
    filter_emotions_by_threshold,
    filter_negative_emotions,
    analyze_hesitancy_vocables,
    get_top_emotions
)

# Import audio concatenation functionality
sys.path.append(str(Path(__file__).parent.parent / "audio_processing"))
from audio_concatenator import AudioConcatenator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
LOGGER = logging.getLogger(__name__)

class ConfigDrivenHumeAnalyzer:
    """
    Config-driven Hume AI analyzer that processes segments based on audio_analysis.json
    and provides iteration_1 compatible output format.
    """
    
    def __init__(self, config_path: str = "config/analysis_config/audio_analysis.json"):
        """Initialize the analyzer with configuration."""
        self.config_path = config_path
        self.config = load_config(config_path)
        self.hume_config = self.config.get('hume_ai', {})
        self.file_paths = self.config.get('file_paths', {})
        
        # Analysis parameters from shared config utility
        analysis_config = get_analysis_config(self.config)
        self.score_threshold = analysis_config['score_threshold']
        self.top_n_emotions = analysis_config['top_n_emotions']
        self.negative_emotions = analysis_config['negative_emotions']
        self.hesitancy_vocables = analysis_config['hesitancy_vocables']
        
        # Extract analysis type from config
        self.analysis_type = self._determine_analysis_type()
        
        # File paths - determine project root correctly
        config_path_obj = Path(config_path)
        if config_path_obj.is_absolute():
            # Find the current_state directory by walking up the path
            current_config_dir = config_path_obj.parent
            while current_config_dir.name != 'current_state' and current_config_dir.parent != current_config_dir:
                current_config_dir = current_config_dir.parent
            
            if current_config_dir.name == 'current_state':
                config_dir = current_config_dir
            else:
                config_dir = Path.cwd()
        else:
            # Relative path - assume we're in project root
            config_dir = Path.cwd()
        
        self.processed_audio_dir = config_dir / self.file_paths.get('processed_audio_dir', 'data/cache/audio/processed')
        self.hume_analysis_dir = config_dir / self.file_paths.get('hume_analysis_dir', 'data/cache/hume_AI_analysis')
        
        # Base output directory - specific campaign directories will be created as needed
        self.hume_analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Hume client with proper config path
        if not Path(config_path).is_absolute():
            config_path = str(config_dir / config_path)
        self.hume_client_config = HumeConfig.from_config_file(config_path)
        self.hume_client = HumeClient(self.hume_client_config)
        
        # Default granularity
        granularity_str = self.hume_config.get('default_granularity', 'utterance')
        self.default_granularity = GranularityLevel(granularity_str)
        
        # Initialize audio concatenator
        self.audio_concatenator = AudioConcatenator(verbose=True)
    
    def _validate_selection_consistency(self, analysis_selection: Dict) -> bool:
        """
        Validate that the analysis selection is consistent with centralized selection_config.json.
        
        Args:
            analysis_selection: The analysis selection from audio_analysis.json
            
        Returns:
            True if selections are consistent, False otherwise
        """
        try:
            # Get centralized selection
            centralized_reps = get_selected_sales_reps()
            centralized_display_names = set(get_sales_rep_display_names())
            
            # Check each sales rep in analysis_selection
            for campaign, campaign_data in analysis_selection.items():
                # Skip documentation keys that start with underscore
                if campaign.startswith('_'):
                    continue
                    
                # Map campaign names
                if campaign == 'monash':
                    centralized_campaign = 'monash_reps'
                elif campaign == 'sol':
                    centralized_campaign = 'sol_reps'
                else:
                    centralized_campaign = f"{campaign}_reps"
                
                for sales_rep_name in campaign_data.keys():
                    # Check if this sales rep is in centralized selection
                    if sales_rep_name not in centralized_display_names:
                        LOGGER.warning(f"Sales rep '{sales_rep_name}' in analysis_selection but not in centralized selection")
                        return False
                    
                    # Check if this sales rep is in the right campaign
                    found_in_campaign = False
                    centralized_campaign_reps = centralized_reps.get(centralized_campaign, [])
                    for rep in centralized_campaign_reps:
                        rep_display_name = rep.get('display_name', f"{rep['firstName']}_{rep['lastName']}")
                        if rep_display_name == sales_rep_name:
                            found_in_campaign = True
                            break
                    
                    if not found_in_campaign:
                        LOGGER.warning(f"Sales rep '{sales_rep_name}' not found in centralized campaign '{centralized_campaign}'")
                        return False
            
            LOGGER.info("✅ Analysis selection validated against centralized config")
            return True
            
        except Exception as e:
            LOGGER.error(f"Error validating selection consistency: {e}")
            return False
        
    
    def _determine_analysis_type(self) -> str:
        """
        Determine the analysis type from config and convert to directory-friendly name.
        
        Returns:
            Analysis type string for directory/file naming
        """
        conversation_analysis = self.hume_config.get('conversation_analysis', {})
        mode = conversation_analysis.get('mode', 'unknown')
        
        # Handle new unified config structure
        if mode == 'segments':
            segment_count = conversation_analysis.get('segment_count', 1)
            analysis_type = f'segments_{segment_count}'
            LOGGER.info(f"📊 Analysis type detected: {analysis_type} (segments mode with {segment_count} segments)")
            return analysis_type
        elif mode == 'full':
            analysis_type = 'full_conversation'
            LOGGER.info(f"📊 Analysis type detected: {analysis_type} (full conversation mode)")
            return analysis_type
        else:
            # Legacy support for old config format
            mode_mapping = {
                'random_1': 'segments_1',
                'random_n': f'segments_{conversation_analysis.get("random_chunk_count", 3)}', 
                'full': 'full_conversation'
            }
            
            analysis_type = mode_mapping.get(mode, f'mode_{mode}')
            LOGGER.info(f"📊 Analysis type detected: {analysis_type} (legacy mode: {mode})")
            return analysis_type
    
    def get_segment_paths(self, campaign: str, sales_rep: str, contact: str, segments: List[str]) -> List[Path]:
        """
        Resolve segment file paths based on the directory structure.
        For conversation analysis mode, looks in conversations/ subdirectory.
        For segments analysis mode, looks in segments/ subdirectory.
        
        Args:
            campaign: Campaign name (e.g., 'monash', 'sol')
            sales_rep: Sales rep name
            contact: Contact identifier
            segments: List of segment filenames or ["ALL"] to auto-discover
            
        Returns:
            List of resolved Path objects for existing segments
        """
        segment_paths = []
        # Fix sales_rep format: replace \ with _ for proper path format
        formatted_sales_rep = sales_rep.replace('\\', '_')
        base_path = self.processed_audio_dir / campaign / formatted_sales_rep / contact
        
        # Determine analysis mode and target directory
        analysis_mode = self.hume_config.get('analysis_mode', 'segments')
        if analysis_mode == 'conversation':
            # For conversation analysis, look in conversations subdirectory
            target_path = base_path / 'conversations'
            LOGGER.info(f"🎙️  Conversation analysis mode - looking in: {target_path}")
        else:
            # For segments analysis, look in segments subdirectory
            target_path = base_path / 'segments'
            LOGGER.info(f"📊 Segments analysis mode - looking in: {target_path}")
        
        # Check if "ALL" option is specified
        if len(segments) == 1 and segments[0].upper() == "ALL":
            LOGGER.info(f"Auto-discovering all files in: {target_path}")
            
            # First try the target subdirectory (conversations/ or segments/)
            search_paths = [target_path]
            
            # If subdirectory doesn't exist, also try the parent contact directory
            if not target_path.exists():
                LOGGER.info(f"Subdirectory not found, trying parent directory: {base_path}")
                search_paths.append(base_path)
            
            for search_path in search_paths:
                if search_path.exists():
                    LOGGER.info(f"Searching for audio files in: {search_path}")
                    # Find all audio files in the directory
                    audio_extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.mp4']
                    all_files = []
                    
                    for ext in audio_extensions:
                        all_files.extend(search_path.glob(f"*{ext}"))
                        all_files.extend(search_path.glob(f"*{ext.upper()}"))
                    
                    # Sort files naturally
                    all_files.sort(key=lambda x: x.name)
                    
                    for file_path in all_files:
                        segment_paths.append(file_path)
                        LOGGER.info(f"Auto-discovered file: {file_path.name}")
                    
                    if segment_paths:
                        # Found files, break out of search loop
                        break
                        
            if not segment_paths:
                LOGGER.warning(f"No audio files found in: {target_path} or {base_path}")
        else:
            # Process specific segments/conversations as before
            for segment in segments:
                if analysis_mode == 'conversation':
                    segment_path = target_path / segment
                else:
                    segment_path = target_path / segment
                
                if segment_path.exists():
                    segment_paths.append(segment_path)
                    LOGGER.info(f"Found file: {segment_path}")
                else:
                    LOGGER.warning(f"File not found: {segment_path}")
        
        return segment_paths
    
    def combine_segments(self, segment_paths: List[Path], combination_method: str = "ordered") -> List[Path]:
        """
        Combine segments based on the specified method.
        
        Args:
            segment_paths: List of segment file paths
            combination_method: Method to combine ('ordered', 'random', 'specific')
            
        Returns:
            List of segment paths in the desired order
        """
        if not segment_paths:
            return []
        
        if combination_method == "random":
            combined = segment_paths.copy()
            random.shuffle(combined)
            LOGGER.info(f"Segments randomized: {[p.name for p in combined]}")
        elif combination_method == "ordered":
            combined = segment_paths
            LOGGER.info(f"Segments in config order: {[p.name for p in combined]}")
        elif combination_method == "specific":
            # For specific, we just use the segments as provided
            combined = segment_paths
            LOGGER.info(f"Specific segments selected: {[p.name for p in combined]}")
        else:
            LOGGER.warning(f"Unknown combination method '{combination_method}', using ordered")
            combined = segment_paths
        
        return combined
    
    async def process_segments(self, segment_paths: List[Path], campaign: str, sales_rep: str, contact: str) -> Optional[Dict[str, Any]]:
        """
        Process segments through Hume AI and return combined results.
        Uses audio concatenation to create a single conversation file for unified analysis.
        
        Args:
            segment_paths: List of segment file paths to process
            campaign: Campaign name
            sales_rep: Sales rep name (FirstName_LastName format)
            contact: Contact identifier
            
        Returns:
            Dictionary containing combined analysis results
        """
        if not segment_paths:
            LOGGER.error("No valid segment paths provided")
            return None
        
        try:
            # Create analysis name for display purposes
            analysis_name = f"{campaign}_{sales_rep}_{contact}"
            
            LOGGER.info(f"Creating conversation file from {len(segment_paths)} segments: {analysis_name}")
            
            # Create concatenated conversation file with metadata
            conversation_file, segment_metadata = self.audio_concatenator.create_conversation_file(
                campaign=campaign,
                sales_rep=sales_rep, 
                contact=contact,
                segment_paths=segment_paths
            )
            
            if not conversation_file or not conversation_file.exists():
                LOGGER.error("Failed to create conversation file")
                return None
            
            LOGGER.info(f"Created conversation file: {conversation_file}")
            
            # Submit the single conversation file to Hume AI
            job_id = await self.hume_client.submit_files([str(conversation_file)], self.default_granularity)
            LOGGER.info(f"Job submitted with ID: {job_id}")
            
            # Wait for job completion
            if await self.hume_client.wait_for_job(job_id):
                LOGGER.info("Job completed successfully")
                
                # Get predictions
                predictions = await self.hume_client.get_job_predictions(job_id, format="json")
                
                # Create organized directory structure and save raw predictions
                # Create campaign-specific directory structure
                if campaign == "monash":
                    campaign_dir = self.hume_analysis_dir / "monash_reps"
                elif campaign == "sol":
                    campaign_dir = self.hume_analysis_dir / "sol_reps"
                else:
                    campaign_dir = self.hume_analysis_dir / f"{campaign}_reps"
                
                # Create full directory path: campaign_reps/FirstName_LastName/contact_id/analysis_type/
                # Sales rep is already in FirstName_LastName format, so use it directly
                output_dir = campaign_dir / sales_rep / contact / self.analysis_type
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create filename: contact_123_analysis_type_analysis_results.json
                predictions_file = output_dir / f"{contact}_{self.analysis_type}_analysis_results.json"
                
                with open(predictions_file, 'w', encoding='utf-8') as f:
                    json.dump(predictions, f, indent=2)
                LOGGER.info(f"Raw predictions saved to: {predictions_file}")
                
                # Parse predictions (now from single conversation file)
                combined_predictions = self.parse_combined_predictions(predictions)
                
                # Get conversation duration
                conversation_duration = self.audio_concatenator.get_conversation_duration(conversation_file)
                
                return {
                    'job_id': job_id,
                    'analysis_name': analysis_name,
                    'segment_count': len(segment_paths),
                    'segment_names': [p.name for p in segment_paths],
                    'segment_metadata': segment_metadata,
                    'conversation_file': str(conversation_file),
                    'conversation_duration': conversation_duration,
                    'predictions': combined_predictions,
                    'raw_predictions_file': str(predictions_file)
                }
            else:
                LOGGER.error("Job failed")
                return None
                
        except Exception as e:
            LOGGER.error(f"Error processing segments: {e}", exc_info=True)
            return None
    
    def parse_combined_predictions(self, predictions: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Parse Hume AI predictions from a single conversation file.
        
        Args:
            predictions: Raw predictions from Hume AI (can be list or dict)
            
        Returns:
            List of time-based predictions from the conversation
        """
        try:
            all_predictions = []
            
            # Handle both list and dict response formats
            if isinstance(predictions, list):
                # API returns a list of prediction objects
                predictions_list = predictions
            else:
                # Fallback for dict format
                predictions_list = predictions.get("results", {}).get("predictions", [])
            
            for file_prediction in predictions_list:
                # Each prediction object has results.predictions
                results = file_prediction.get("results", {})
                file_predictions_list = results.get("predictions", [])
                
                # Check for errors
                errors = results.get("errors", [])
                if errors:
                    for error in errors:
                        LOGGER.warning(f"API Error for file {error.get('file', 'unknown')}: {error.get('message', 'Unknown error')}")
                
                for file_pred in file_predictions_list:
                    models = file_pred.get("models", {})
                    
                    # Try burst model first, then prosody, then language
                    for model_name in ["burst", "prosody", "language"]:
                        model_data = models.get(model_name, {})
                        model_grouped = model_data.get("grouped_predictions", [])
                        
                        if model_grouped:
                            file_predictions = model_grouped[0].get("predictions", [])
                            if file_predictions:
                                break
                    else:
                        # No predictions found in any model
                        continue
                
                    # Add all predictions from this conversation file (no time offset needed)
                    all_predictions.extend(file_predictions)
            
            LOGGER.info(f"Extracted {len(all_predictions)} predictions from conversation file")
            return all_predictions
            
        except Exception as e:
            LOGGER.error(f"Error parsing conversation predictions: {e}", exc_info=True)
            return []
    
    def analyze_predictions_iteration1_format(self, predictions: List[Dict[str, Any]], analysis_name: str, 
                                             segment_metadata: List[Dict[str, Any]] = None):
        """
        Analyze predictions and display results in iteration_1 format with original timestamps.
        
        Args:
            predictions: List of time-based predictions
            analysis_name: Name of the analysis for display
            segment_metadata: List of segment metadata with original timestamps
        """
        if not predictions:
            LOGGER.error("No predictions to analyze")
            return
        
        # Helper function to map concatenated timestamps to original conversation timestamps
        def map_to_original_timestamp(concatenated_time: float) -> tuple[float, str]:
            """
            Map a concatenated timestamp to original conversation timestamp.
            
            Returns:
                Tuple of (original_time, display_string)
            """
            if not segment_metadata:
                return concatenated_time, f"{concatenated_time:.1f}s"
            
            # Find which segment this timestamp belongs to
            for segment in segment_metadata:
                concat_start = segment['concatenated_start_seconds']
                concat_end = segment['concatenated_end_seconds']
                
                if concat_start <= concatenated_time <= concat_end:
                    # Calculate offset within the segment
                    offset_in_segment = concatenated_time - concat_start
                    original_time = segment['original_start_seconds'] + offset_in_segment
                    return original_time, f"{concatenated_time:.1f}s (orig: {original_time:.1f}s)"
            
            # Fallback if not found in any segment
            return concatenated_time, f"{concatenated_time:.1f}s"
        
        # Extract emotions and descriptions using shared utilities
        all_emotions = extract_emotions_from_predictions(predictions)
        all_descriptions = extract_descriptions_from_predictions(predictions)
        
        if not all_emotions:
            LOGGER.error("No emotion data found in predictions")
            return
        
        # Calculate total duration
        total_duration_seconds = max((pred['time']['end'] for pred in predictions), default=0)
        
        # Convert to DataFrames for analysis using shared utilities
        emotions_df = create_emotions_dataframe(all_emotions)
        descriptions_df = create_descriptions_dataframe(all_descriptions)
        
        # Display analysis header
        LOGGER.info("\\n" + "="*60)
        LOGGER.info(f"🎭 HUME AI ANALYSIS RESULTS: {analysis_name}")
        LOGGER.info("="*60)
        LOGGER.info(f"📊 Total Duration: {total_duration_seconds:.2f} seconds ({total_duration_seconds/60:.2f} minutes)")
        LOGGER.info(f"🎯 Emotion Data Points: {len(all_emotions)}")
        LOGGER.info(f"🗣️  Description Data Points: {len(all_descriptions)}")
        
        # Display segment mapping information if available
        if segment_metadata:
            LOGGER.info("\\n" + "="*50)
            LOGGER.info("📍 SEGMENT MAPPING")
            LOGGER.info("="*50)
            LOGGER.info(f"🎯 Analysis includes {len(segment_metadata)} segments from original conversation:")
            for segment in segment_metadata:
                orig_start = segment['original_start_seconds']
                orig_end = segment['original_end_seconds']
                concat_start = segment['concatenated_start_seconds']
                concat_end = segment['concatenated_end_seconds']
                LOGGER.info(f"   Segment {segment['segment_number']}: "
                          f"Original {orig_start:.1f}s-{orig_end:.1f}s → "
                          f"Analysis {concat_start:.1f}s-{concat_end:.1f}s "
                          f"({segment['source_file']})")
        
        # 1. Average Emotion Scores using shared utility
        LOGGER.info("\\n" + "="*50)
        LOGGER.info(f"Top {self.top_n_emotions} Average Emotion Scores")
        LOGGER.info("="*50)
        avg_scores = get_top_emotions(emotions_df, n=self.top_n_emotions, aggregation='mean')
        LOGGER.info(avg_scores.to_string())
        
        # 2. Peak Emotion Moments (iteration_1 format) with original timestamps
        LOGGER.info("\\n" + "="*50)
        LOGGER.info(f"Top Peak Emotion Moments (Score > {self.score_threshold})")
        LOGGER.info("="*50)
        peak_emotions = filter_emotions_by_threshold(emotions_df, self.score_threshold).sort_values(
            by='score', ascending=False
        ).head(20)
        if peak_emotions.empty:
            LOGGER.info("No peak emotion moments found above the threshold.")
        else:
            # Enhanced display with original timestamps if available
            if segment_metadata:
                LOGGER.info("Peak emotions with original conversation timestamps:")
                for _, row in peak_emotions.iterrows():
                    orig_time, time_display = map_to_original_timestamp(row['time'])
                    LOGGER.info(f"   {time_display} - {row['emotion']}: {row['score']:.3f}")
            else:
                LOGGER.info(peak_emotions.to_string())
        
        # 3. Negative Emotion Analysis (iteration_1 format)
        LOGGER.info("\\n" + "="*50)
        LOGGER.info("Negative Emotion Analysis")
        LOGGER.info("="*50)
        negative_df = filter_negative_emotions(emotions_df, self.negative_emotions, self.score_threshold)
        negative_event_count = len(negative_df)
        total_windows = len(emotions_df['time'].unique())
        
        if total_windows > 0:
            negative_rate = (negative_event_count / total_windows) * 100
            LOGGER.info(f"Significant negative emotions were detected in {negative_event_count} of {total_windows} time windows ({negative_rate:.2f}%).")
            if not negative_df.empty:
                LOGGER.info(f"Most frequent negative emotions:\\n{negative_df['emotion'].value_counts().nlargest(5).to_string()}")
        else:
            LOGGER.info("No time windows to analyze for negative emotions.")
        
        # 4. Hesitancy Vocables Analysis using shared utility
        LOGGER.info("\\n" + "="*50)
        LOGGER.info("Hesitancy Vocable Analysis")
        LOGGER.info("="*50)
        hesitancy_analysis = analyze_hesitancy_vocables(descriptions_df, self.hesitancy_vocables, self.score_threshold)
        total_hesitancy_events = hesitancy_analysis['total_count']
        total_minutes = total_duration_seconds / 60
        
        if total_minutes > 0:
            hesitancy_per_minute = total_hesitancy_events / total_minutes
            LOGGER.info(f"Total significant hesitancy vocables: {total_hesitancy_events}")
            LOGGER.info(f"Total call duration: {total_minutes:.2f} minutes")
            LOGGER.info(f"Average hesitancy events per minute: {hesitancy_per_minute:.2f}")
            if hesitancy_analysis['vocable_counts']:
                vocable_counts_str = '\\n'.join(f"{k}: {v}" for k, v in hesitancy_analysis['vocable_counts'].items())
                LOGGER.info(f"Most frequent hesitancy vocables:\\n{vocable_counts_str}")
        else:
            LOGGER.info("Not enough duration to calculate per-minute hesitancy rates.")
        
        LOGGER.info("\\n" + "="*60)
    
    async def update_career_analysis(self, processed_sales_reps: set):
        """
        Update career analysis for processed sales reps.
        
        Args:
            processed_sales_reps: Set of (campaign, sales_rep) tuples
        """
        if not processed_sales_reps:
            return
        
        LOGGER.info("\\n" + "="*60)
        LOGGER.info("🔄 UPDATING CAREER ANALYSIS")
        LOGGER.info("="*60)
        
        # Import career analysis functionality
        try:
            from hume.career_aggregated_analysis import process_with_career_config
        except ImportError:
            LOGGER.error("Could not import career analysis functionality")
            return
        
        # Group sales reps by campaign
        campaigns_to_reps = {}
        for campaign, sales_rep in processed_sales_reps:
            if campaign not in campaigns_to_reps:
                campaigns_to_reps[campaign] = []
            campaigns_to_reps[campaign].append(sales_rep)
        
        # Update career analysis for each affected sales rep
        for campaign, sales_reps in campaigns_to_reps.items():
            LOGGER.info(f"🏢 Updating career analysis for {campaign} campaign")
            LOGGER.info(f"👥 Sales reps: {', '.join(sales_reps)}")
            
            # Create a temporary career config for these specific sales reps
            temp_career_config = {
                'selection_criteria': {
                    'sales_reps': {
                        'selected': sales_reps
                    },
                    'analysis_modes': {
                        'selected': ['all']  # Include all analysis modes
                    }
                },
                'analysis_parameters': {
                    'emotion_analysis': {
                        'score_threshold': 0.6,
                        'peak_emotion_threshold': 0.6,
                        'top_n_emotions': 10
                    },
                    'hesitancy_analysis': {
                        'score_threshold': 0.1,
                        'vocables': ["Uh", "Ugh", "Uh-huh", "Hmm", "Mhm", "Mmm", "Umm"]
                    },
                    'negative_emotion_analysis': {
                        'score_threshold': 0.6,
                        'negative_emotions': [
                            "Anger", "Anxiety", "Awkwardness", "Boredom", "Confusion", "Contempt",
                            "Disgust", "Distress", "Disappointment", "Doubt", "Embarrassment",
                            "Empathic Pain", "Fear", "Guilt", "Horror", "Pain", "Sadness", "Shame",
                            "Surprise (negative)", "Tiredness"
                        ]
                    }
                },
                'directory_settings': {
                    'hume_analysis_base_path': 'data/cache/hume_AI_analysis',
                    'use_relative_paths': True,
                    'auto_detect_campaigns': True
                },
                'validation_and_error_handling': {
                    'strict_mode': False,
                    'minimum_files_required': 1,
                    'missing_sales_rep_action': 'warn_and_skip',
                    'missing_analysis_mode_action': 'warn_and_continue',
                    'validation_rules': {
                        'require_emotion_data': True,
                        'require_minimum_duration': False,
                        'minimum_duration_seconds': 30
                    }
                },
                'output_settings': {
                    'verbose_logging': True,
                    'log_level': 'INFO',
                    'show_progress': True,
                    'save_processing_summary': False
                }
            }
            
            try:
                # Create temporary config file 
                import json
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_file:
                    json.dump(temp_career_config, temp_file, indent=2)
                    temp_config_path = temp_file.name
                
                # Run career analysis
                result = process_with_career_config(temp_config_path)
                if result['success']:
                    LOGGER.info(f"✅ Career analysis updated successfully for {len(sales_reps)} sales reps")
                    summary = result.get('summary', {})
                    LOGGER.info(f"📊 Processed: {summary.get('successful', 0)}/{summary.get('total', 0)} sales reps")
                else:
                    LOGGER.error(f"❌ Career analysis update failed: {result.get('error', 'Unknown error')}")
                
                # Clean up temporary file
                import os
                try:
                    os.unlink(temp_config_path)
                except:
                    pass
                    
            except Exception as e:
                LOGGER.error(f"Error updating career analysis: {e}")
        
        LOGGER.info("🔄 Career analysis update complete")
        LOGGER.info("="*60)
    
    async def run_analysis(self):
        """
        Run the complete config-driven analysis process.
        """
        LOGGER.info("🚀 Starting Config-Driven Hume AI Analysis")
        LOGGER.info(f"📁 Processed audio directory: {self.processed_audio_dir}")
        LOGGER.info(f"📁 Analysis output directory: {self.hume_analysis_dir}")
        
        # Get analysis selections from config
        analysis_selection = self.hume_config.get('analysis_selection', {})
        if not analysis_selection:
            LOGGER.error("No analysis_selection found in audio_analysis.json")
            return
        
        # Validate against centralized selection
        if not self._validate_selection_consistency(analysis_selection):
            LOGGER.error("Analysis selection is inconsistent with centralized selection_config.json")
            return
        
        # Get combination method from config (default to ordered)
        combination_method = self.hume_config.get('combination_method', 'ordered')
        
        # Track processed sales reps for career analysis updates
        processed_sales_reps = set()
        
        # Process each campaign
        for campaign, campaign_data in analysis_selection.items():
            # Skip documentation keys that start with underscore
            if campaign.startswith('_'):
                continue
                
            LOGGER.info(f"\\n🏢 Processing campaign: {campaign}")
            
            for sales_rep, rep_data in campaign_data.items():
                LOGGER.info(f"👤 Processing sales rep: {sales_rep}")
                
                for contact, contact_data in rep_data.items():
                    LOGGER.info(f"📞 Processing contact: {contact}")
                    
                    segments = contact_data.get('segments', [])
                    if not segments:
                        LOGGER.warning(f"No segments specified for {campaign}/{sales_rep}/{contact}")
                        continue
                    
                    # Resolve segment paths
                    segment_paths = self.get_segment_paths(campaign, sales_rep, contact, segments)
                    if not segment_paths:
                        LOGGER.warning(f"No valid segments found for {campaign}/{sales_rep}/{contact}")
                        continue
                    
                    # Combine segments based on method
                    combined_segments = self.combine_segments(segment_paths, combination_method)
                    
                    # Process segments
                    results = await self.process_segments(combined_segments, campaign, sales_rep, contact)
                    if results:
                        # Analyze and display results in iteration_1 format with segment metadata
                        self.analyze_predictions_iteration1_format(
                            results['predictions'], 
                            results['analysis_name'],
                            results.get('segment_metadata', [])
                        )
                        
                        # Track this sales rep for career analysis update
                        processed_sales_reps.add((campaign, sales_rep))
                    else:
                        analysis_name = f"{campaign}_{sales_rep}_{contact}"
                        LOGGER.error(f"Failed to process segments for {analysis_name}")
        
        # Update career analysis for all processed sales reps
        await self.update_career_analysis(processed_sales_reps)

async def main():
    """Main entry point for config-driven analysis."""
    try:
        analyzer = ConfigDrivenHumeAnalyzer()
        await analyzer.run_analysis()
        LOGGER.info("\\n✅ Config-driven analysis completed successfully!")
    except Exception as e:
        LOGGER.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())