#!/usr/bin/env python3
"""
Career Aggregated Analysis Script

Aggregates all Hume AI analysis results across different analysis types 
(random_single, random_multi, full_conversation) for each sales rep to 
create comprehensive career analysis reports.

Generates career_analysis.json files stored in:
hume_AI_analysis/{campaign}_reps/{sales_rep}/career/career_analysis.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
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

from hume.core.config_loader import load_config
from hume.core.analysis_config import get_analysis_config
from hume.core.prediction_parser import parse_prediction_file_with_duration
from hume.core.emotion_processor import (
    extract_emotions_from_predictions,
    extract_descriptions_from_predictions,
    create_emotions_dataframe,
    create_descriptions_dataframe,
    calculate_emotion_statistics
)

# For backward compatibility, alias the new function to the old name
parse_prediction_file = parse_prediction_file_with_duration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
LOGGER = logging.getLogger(__name__)

def load_career_config(career_config_path: str = None) -> dict:
    """Load career-specific configuration from career_config.json file."""
    try:
        if career_config_path:
            config_file = Path(career_config_path)
        else:
            # Default location
            script_dir = Path(__file__).parent.parent.parent
            config_file = script_dir / "config" / "analysis_config" / "career_config.json"
        
        if not config_file.exists():
            LOGGER.warning(f"Career config file not found at {config_file}. Using default settings.")
            return {}
            
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            LOGGER.info(f"Loaded career configuration from: {config_file}")
            return config
    except Exception as e:
        LOGGER.error(f"Error loading career config: {e}")
        return {}



def find_analysis_files(sales_rep_dir: Path, selected_modes: List[str] = None) -> List[Path]:
    """Find analysis JSON files for a sales rep, optionally filtered by analysis types."""
    json_files = []
    
    # Default to all analysis types if none specified
    if selected_modes is None or 'all' in selected_modes:
        # Support both legacy and new analysis types
        analysis_types = ['random_single', 'random_multi', 'full_conversation', 'segments', 'full']
    else:
        analysis_types = selected_modes
    
    # Find all JSON files with analysis results in the entire sales rep directory tree
    all_analysis_files = list(sales_rep_dir.glob("**/*_analysis_results.json"))
    
    # Filter based on analysis types if specific types are requested
    if 'all' not in (selected_modes or []):
        filtered_files = []
        for file_path in all_analysis_files:
            parent_dir_name = file_path.parent.name
            
            # Check for legacy patterns
            if any(pattern in parent_dir_name for pattern in ['random_single', 'random_multi', 'full_conversation']):
                if any(legacy_type in parent_dir_name for legacy_type in analysis_types):
                    filtered_files.append(file_path)
            
            # Check for new patterns: segments_N/ and full_conversation/
            elif parent_dir_name.startswith('segments_') and ('segments' in analysis_types):
                filtered_files.append(file_path)
            elif parent_dir_name == 'full_conversation' and ('full_conversation' in analysis_types or 'full' in analysis_types):
                filtered_files.append(file_path)
            
            # Also include files directly in contact directories (legacy support)
            elif parent_dir_name.startswith('contact_') and file_path.name.endswith('_analysis_results.json'):
                filtered_files.append(file_path)
        
        json_files = filtered_files
    else:
        json_files = all_analysis_files
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for file_path in json_files:
        if file_path not in seen:
            seen.add(file_path)
            unique_files.append(file_path)
    
    LOGGER.info(f"Found {len(unique_files)} total analysis files for {sales_rep_dir.name}")
    LOGGER.debug(f"Analysis files found: {[f.relative_to(sales_rep_dir) for f in unique_files]}")
    
    return unique_files

def resolve_campaigns(hume_analysis_dir: Path, selected_campaigns: List[str]) -> List[Path]:
    """Resolve campaign directories based on selection criteria."""
    available_campaigns = [d for d in hume_analysis_dir.iterdir() if d.is_dir() and d.name.endswith('_reps')]
    
    if not available_campaigns:
        LOGGER.warning("No campaign directories found (directories ending with '_reps')")
        return []
    
    if 'all' in selected_campaigns:
        LOGGER.info(f"Auto-detected {len(available_campaigns)} campaigns: {[c.name for c in available_campaigns]}")
        return available_campaigns
    else:
        resolved_campaigns = []
        for campaign_name in selected_campaigns:
            campaign_dir = hume_analysis_dir / campaign_name
            if campaign_dir.exists() and campaign_dir.is_dir():
                resolved_campaigns.append(campaign_dir)
            else:
                LOGGER.warning(f"Campaign directory not found: {campaign_name}")
        return resolved_campaigns

def resolve_sales_reps(campaign_dirs: List[Path], selected_sales_reps: List[str]) -> List[Path]:
    """Resolve sales rep directories based on selection criteria."""
    all_sales_rep_dirs = []
    
    # Collect all available sales rep directories
    for campaign_dir in campaign_dirs:
        sales_rep_dirs = [d for d in campaign_dir.iterdir() if d.is_dir()]
        all_sales_rep_dirs.extend(sales_rep_dirs)
    
    if not all_sales_rep_dirs:
        LOGGER.warning("No sales rep directories found")
        return []
    
    if 'all' in selected_sales_reps:
        LOGGER.info(f"Processing all {len(all_sales_rep_dirs)} sales reps")
        return all_sales_rep_dirs
    else:
        resolved_sales_reps = []
        available_names = {d.name: d for d in all_sales_rep_dirs}
        
        for sales_rep_name in selected_sales_reps:
            if sales_rep_name in available_names:
                resolved_sales_reps.append(available_names[sales_rep_name])
            else:
                LOGGER.warning(f"Sales rep not found: {sales_rep_name}. Available: {list(available_names.keys())}")
        
        return resolved_sales_reps

def validate_configuration(career_config: dict) -> bool:
    """Validate career configuration and log any issues."""
    valid = True
    
    # Check selection criteria
    selection = career_config.get('selection_criteria', {})
    campaigns = selection.get('campaigns', {}).get('selected', [])
    sales_reps = selection.get('sales_reps', {}).get('selected', [])
    
    if not campaigns and not sales_reps:
        LOGGER.error("No campaigns or sales reps selected in configuration")
        valid = False
    
    if campaigns and sales_reps:
        LOGGER.warning("Both campaigns and sales_reps specified. Using sales_reps (campaigns will be ignored)")
    
    # Check analysis modes
    analysis_modes = selection.get('analysis_modes', {}).get('selected', [])
    valid_modes = ['random_single', 'random_multi', 'full_conversation', 'segments', 'full', 'all']
    
    for mode in analysis_modes:
        if mode not in valid_modes:
            LOGGER.error(f"Invalid analysis mode: {mode}. Valid modes: {valid_modes}")
            valid = False
    
    return valid

def merge_analysis_parameters(base_config: dict, custom_params: dict = None) -> dict:
    """Merge custom parameters with base analysis configuration."""
    if custom_params is None:
        return base_config
        
    merged_config = base_config.copy()
    
    # Deep merge custom parameters
    for section, params in custom_params.items():
        if section in merged_config and isinstance(params, dict):
            merged_config[section].update(params)
        else:
            merged_config[section] = params
    
    return merged_config

def aggregate_rep_data(json_files: List[Path], analysis_config: dict) -> Dict[str, Any]:
    """
    Aggregate all analysis data for a single sales rep across all contacts and analysis types.
    """
    all_emotions = []
    all_descriptions = []
    total_duration_seconds = 0.0
    file_count = 0
    
    score_threshold = analysis_config['score_threshold']
    peak_threshold = analysis_config['peak_emotion_threshold']
    negative_emotions = analysis_config['negative_emotions']
    hesitancy_vocables = analysis_config['hesitancy_vocables']
    
    # Process each analysis file
    for file_path in json_files:
        predictions, duration = parse_prediction_file(file_path)
        if not predictions:
            continue
            
        file_count += 1
        total_duration_seconds += duration
        
        # Extract emotions and descriptions from this file
        for pred in predictions:
            time_s = pred.get('time', {}).get('begin', 0)
            
            # Collect emotions
            for emotion in pred.get('emotions', []):
                # Handle missing keys gracefully
                name = emotion.get('name', 'unknown')
                score = emotion.get('score', 0.0)
                
                all_emotions.append({
                    'file': file_path.name,
                    'time': time_s,
                    'emotion': name,
                    'score': score
                })
            
            # Collect descriptions (vocables) with defensive programming
            for desc in pred.get('descriptions', []):
                # Handle both possible key names and missing keys
                name = desc.get('name') or desc.get('description', 'unknown')
                score = desc.get('score', 0.0)
                
                all_descriptions.append({
                    'file': file_path.name,
                    'time': time_s,
                    'description': name,
                    'score': score
                })
    
    if not all_emotions:
        LOGGER.warning("No emotion data found for this sales rep")
        return {}
    
    # Convert to DataFrames for analysis
    emotions_df = pd.DataFrame(all_emotions)
    descriptions_df = pd.DataFrame(all_descriptions)
    
    # Calculate emotion metrics
    emotion_summary = calculate_emotion_metrics(emotions_df, analysis_config)
    
    # Calculate top negative emotions specifically
    top_negative_summary = calculate_top_negative_emotions(emotions_df, analysis_config, top_n=5)
    
    # Calculate hesitancy metrics
    hesitancy_summary = calculate_hesitancy_metrics(descriptions_df, analysis_config, file_count)
    
    # Calculate negative emotion metrics
    negative_emotion_summary = calculate_negative_emotion_metrics(emotions_df, analysis_config)
    
    # Create comprehensive career analysis
    career_analysis = {
        'metadata': {
            'total_files_analyzed': file_count,
            'total_duration_seconds': round(total_duration_seconds, 2),
            'total_duration_minutes': round(total_duration_seconds / 60, 2),
            'total_emotion_data_points': len(all_emotions),
            'total_description_data_points': len(all_descriptions),
            'analysis_date': pd.Timestamp.now().isoformat()
        },
        'emotion_analysis': emotion_summary,
        'top_negative_emotions_analysis': top_negative_summary,
        'hesitancy_vocable_analysis': hesitancy_summary,
        'negative_emotion_analysis': negative_emotion_summary
    }
    
    return career_analysis

def calculate_emotion_metrics(emotions_df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """Calculate comprehensive emotion metrics."""
    peak_threshold = config['peak_emotion_threshold']
    top_n = config['top_n_emotions']
    
    # Average emotion scores across all contacts
    avg_emotions = emotions_df.groupby('emotion')['score'].mean().sort_values(ascending=False)
    top_avg_emotions = avg_emotions.head(top_n).to_dict()
    
    # Peak emotion moments (score > 0.6)
    peak_emotions = emotions_df[emotions_df['score'] > peak_threshold].copy()
    peak_moments = []
    
    if not peak_emotions.empty:
        # Get top peak moments
        top_peaks = peak_emotions.nlargest(20, 'score')
        peak_moments = [
            {
                'emotion': row['emotion'],
                'score': round(row['score'], 4),
                'time': round(row['time'], 2),
                'file': row['file']
            }
            for _, row in top_peaks.iterrows()
        ]
    
    return {
        'top_average_emotion_scores': {
            emotion: round(score, 4) for emotion, score in top_avg_emotions.items()
        },
        'peak_emotion_moments': {
            'threshold_used': peak_threshold,
            'total_peak_moments': len(peak_emotions),
            'top_peak_moments': peak_moments
        },
        'emotion_statistics': {
            'unique_emotions_detected': emotions_df['emotion'].nunique(),
            'total_emotion_measurements': len(emotions_df)
        }
    }


def calculate_top_negative_emotions(emotions_df: pd.DataFrame, config: dict, top_n: int = 5) -> Dict[str, Any]:
    """Calculate top N negative emotions specifically for performance analysis."""
    negative_emotions = config['negative_emotions']
    score_threshold = config.get('score_threshold', 0.6)
    
    # Filter for negative emotions only
    negative_df = emotions_df[emotions_df['emotion'].isin(negative_emotions)].copy()
    
    if negative_df.empty:
        return {
            'top_negative_emotions': {},
            'negative_emotions_summary': {
                'total_negative_measurements': 0,
                'unique_negative_emotions_detected': 0,
                'threshold_used': score_threshold,
                'analysis_note': 'No negative emotions detected above threshold'
            }
        }
    
    # Calculate average scores for negative emotions
    negative_avg_scores = negative_df.groupby('emotion')['score'].mean().sort_values(ascending=False)
    top_negative = negative_avg_scores.head(top_n).to_dict()
    
    # Get significant negative emotions (above threshold)
    significant_negative = negative_df[negative_df['score'] > score_threshold]
    
    # Calculate frequency of each negative emotion above threshold
    negative_frequency = significant_negative['emotion'].value_counts().to_dict()
    
    return {
        'top_negative_emotions': {
            emotion: round(score, 4) for emotion, score in top_negative.items()
        },
        'negative_emotions_summary': {
            'total_negative_measurements': len(negative_df),
            'significant_negative_events': len(significant_negative),
            'unique_negative_emotions_detected': negative_df['emotion'].nunique(),
            'threshold_used': score_threshold,
            'most_frequent_negative_above_threshold': negative_frequency
        }
    }

def calculate_per_file_hesitancy_metrics(descriptions_df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """Calculate hesitancy metrics for each individual audio file."""
    hesitancy_vocables = config['hesitancy_vocables']
    score_threshold = config.get('hesitancy_score_threshold', config['score_threshold'])
    
    # Filter for significant hesitancy vocables
    hesitancy_df = descriptions_df[
        descriptions_df['description'].isin(hesitancy_vocables) &
        (descriptions_df['score'] > score_threshold)
    ]
    
    # Group by file and calculate metrics for each file
    per_file_metrics = {}
    file_summary = {}
    
    # Get all unique files from descriptions_df
    all_files = descriptions_df['file'].unique()
    
    for file_name in all_files:
        # Get hesitancy events for this specific file
        file_hesitancy = hesitancy_df[hesitancy_df['file'] == file_name]
        
        # Calculate metrics
        total_events = len(file_hesitancy)
        avg_score = round(file_hesitancy['score'].mean(), 4) if total_events > 0 else 0.0
        
        # Count of each hesitancy vocable type in this file
        vocable_counts = file_hesitancy['description'].value_counts().to_dict()
        
        per_file_metrics[file_name] = {
            'total_hesitancy_events': total_events,
            'average_hesitancy_score': avg_score,
            'hesitancy_vocables_detected': vocable_counts
        }
        
        # For the summary display format
        file_summary[file_name] = avg_score
    
    # Sort files by average hesitancy score (descending)
    sorted_summary = dict(sorted(file_summary.items(), key=lambda x: x[1], reverse=True))
    
    return {
        'detailed_per_file_metrics': per_file_metrics,
        'summary_average_hesitancy_per_file': sorted_summary,
        'files_with_hesitancy': sum(1 for score in file_summary.values() if score > 0),
        'total_files_analyzed': len(all_files)
    }

def calculate_hesitancy_metrics(descriptions_df: pd.DataFrame, config: dict, file_count: int) -> Dict[str, Any]:
    """Calculate hesitancy vocable metrics."""
    hesitancy_vocables = config['hesitancy_vocables']
    # Use specific hesitancy threshold if available, otherwise fall back to general score_threshold
    score_threshold = config.get('hesitancy_score_threshold', config['score_threshold'])
    
    # Filter for significant hesitancy vocables
    hesitancy_df = descriptions_df[
        descriptions_df['description'].isin(hesitancy_vocables) &
        (descriptions_df['score'] > score_threshold)
    ]
    
    total_hesitancy_events = len(hesitancy_df)
    
    # Average hesitancy per audio file
    avg_hesitancy_per_file = total_hesitancy_events / file_count if file_count > 0 else 0
    
    # Most frequent hesitancy vocables
    hesitancy_counts = hesitancy_df['description'].value_counts().to_dict()
    
    # Calculate per-file hesitancy metrics
    per_file_hesitancy = calculate_per_file_hesitancy_metrics(descriptions_df, config)
    
    return {
        'total_hesitancy_events': total_hesitancy_events,
        'average_hesitancy_per_audio_file': round(avg_hesitancy_per_file, 2),
        'hesitancy_vocables_breakdown': hesitancy_counts,
        'threshold_used': score_threshold,
        'per_file_hesitancy_analysis': per_file_hesitancy,
        'analysis_summary': f"Detected {total_hesitancy_events} significant hesitancy events across {file_count} audio files"
    }

def calculate_negative_emotion_metrics(emotions_df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """Calculate negative emotion analysis."""
    negative_emotions = config['negative_emotions']
    score_threshold = config['score_threshold']
    
    # Filter for significant negative emotions
    negative_df = emotions_df[
        emotions_df['emotion'].isin(negative_emotions) &
        (emotions_df['score'] > score_threshold)
    ]
    
    total_negative_events = len(negative_df)
    total_time_windows = len(emotions_df['time'].unique())
    
    # Calculate negative emotion windows (time segments with significant negative emotions)
    negative_windows = len(negative_df['time'].unique())
    negative_rate = (negative_windows / total_time_windows * 100) if total_time_windows > 0 else 0
    
    # Most frequent negative emotions
    frequent_negative = negative_df['emotion'].value_counts().head(10).to_dict()
    
    return {
        'significant_negative_emotion_windows': negative_windows,
        'total_time_windows_analyzed': total_time_windows,
        'negative_emotion_rate_percentage': round(negative_rate, 2),
        'most_frequent_negative_emotions': frequent_negative,
        'total_negative_emotion_events': total_negative_events,
        'threshold_used': score_threshold,
        'analysis_summary': f"Significant negative emotions detected in {negative_windows} of {total_time_windows} time windows ({negative_rate:.2f}%)"
    }

def process_sales_rep(sales_rep_dir: Path, analysis_config: dict, selected_modes: List[str] = None) -> Dict[str, Any]:
    """Process a single sales rep and generate career analysis."""
    LOGGER.info(f"Processing sales rep: {sales_rep_dir.name}")
    
    # Find analysis files based on selected modes
    json_files = find_analysis_files(sales_rep_dir, selected_modes)
    
    if not json_files:
        LOGGER.warning(f"No analysis files found for {sales_rep_dir.name}")
        return {'success': False, 'error': 'No analysis files found'}
    
    # Validate minimum requirements
    validation_rules = analysis_config.get('validation_rules', {})
    min_files = validation_rules.get('minimum_files_required', 1)
    
    if len(json_files) < min_files:
        LOGGER.warning(f"Insufficient files for {sales_rep_dir.name}: {len(json_files)} < {min_files} required")
        return {'success': False, 'error': f'Insufficient files: {len(json_files)} < {min_files} required'}
    
    # Aggregate data across all files
    career_analysis = aggregate_rep_data(json_files, analysis_config)
    
    if not career_analysis:
        LOGGER.warning(f"Could not generate career analysis for {sales_rep_dir.name}")
        return {'success': False, 'error': 'Failed to generate career analysis'}
    
    # Create career directory
    career_dir = sales_rep_dir / 'career'
    career_dir.mkdir(exist_ok=True)
    
    # Save career analysis
    output_file = career_dir / 'career_analysis.json'
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(career_analysis, f, indent=2, ensure_ascii=False)
        
        LOGGER.info(f"Career analysis saved to: {output_file}")
        
        # Log summary
        metadata = career_analysis['metadata']
        LOGGER.info(f"  - Files analyzed: {metadata['total_files_analyzed']}")
        LOGGER.info(f"  - Total duration: {metadata['total_duration_minutes']:.2f} minutes")
        LOGGER.info(f"  - Emotion data points: {metadata['total_emotion_data_points']}")
        
        return {
            'success': True, 
            'output_file': str(output_file),
            'metadata': metadata,
            'sales_rep': sales_rep_dir.name
        }
        
    except Exception as e:
        LOGGER.error(f"Failed to save career analysis for {sales_rep_dir.name}: {e}")
        return {'success': False, 'error': f'Failed to save: {str(e)}'}

def find_hume_analysis_directory() -> Optional[Path]:
    """Find the hume_AI_analysis directory."""
    # Start from current directory and work up
    current_dir = Path.cwd()
    
    # Check current directory and parent directories for standard structure
    for path in [current_dir] + list(current_dir.parents):
        # Check for data/cache/hume_AI_analysis (standard structure)
        hume_analysis_dir = path / 'data' / 'cache' / 'hume_AI_analysis'
        if hume_analysis_dir.exists() and hume_analysis_dir.is_dir():
            return hume_analysis_dir
        
        # Check for direct hume_AI_analysis (legacy structure)
        hume_analysis_dir = path / 'hume_AI_analysis'
        if hume_analysis_dir.exists() and hume_analysis_dir.is_dir():
            return hume_analysis_dir
    
    return None

def process_with_career_config(career_config_path: str = None, base_config_path: str = "config/analysis_config/audio_analysis.json") -> Dict[str, Any]:
    """Process career analysis using career_config.json configuration."""
    
    # Load configurations
    career_config = load_career_config(career_config_path)
    base_config = load_config(base_config_path)
    
    if not career_config:
        LOGGER.error("Could not load career configuration. Using legacy mode.")
        return {'success': False, 'error': 'Career configuration not loaded'}
    
    # Validate configuration
    if not validate_configuration(career_config):
        return {'success': False, 'error': 'Configuration validation failed'}
    
    # Extract configuration sections
    selection_criteria = career_config.get('selection_criteria', {})
    
    # If no specific selection in career_config, use centralized selection
    if not selection_criteria.get('sales_reps', {}).get('selected'):
        try:
            centralized_reps = get_sales_rep_display_names()
            if centralized_reps:
                LOGGER.info(f"Using centralized selection: {centralized_reps}")
                selection_criteria = {
                    'sales_reps': {'selected': centralized_reps},
                    'analysis_modes': {'selected': ['all']}
                }
            else:
                LOGGER.warning("No sales reps found in centralized selection_config.json")
        except Exception as e:
            LOGGER.warning(f"Could not load centralized selection, using career_config.json: {e}")
    else:
        LOGGER.info("Using career_config.json selection (overriding centralized)")
        # Validate against centralized selection
        try:
            centralized_reps = set(get_sales_rep_display_names())
            config_reps = set(selection_criteria.get('sales_reps', {}).get('selected', []))
            invalid_reps = config_reps - centralized_reps
            if invalid_reps:
                LOGGER.warning(f"Sales reps in career_config.json not found in centralized config: {invalid_reps}")
        except Exception as e:
            LOGGER.warning(f"Could not validate against centralized selection: {e}")
    analysis_parameters = career_config.get('analysis_parameters', {})
    batch_config = career_config.get('batch_processing', {})
    directory_settings = career_config.get('directory_settings', {})
    validation_settings = career_config.get('validation_and_error_handling', {})
    output_settings = career_config.get('output_settings', {})
    
    # Setup logging level
    if output_settings.get('verbose_logging', True):
        logging.getLogger().setLevel(getattr(logging, output_settings.get('log_level', 'INFO')))
    
    # Find hume analysis directory
    hume_base_path = directory_settings.get('hume_analysis_base_path', 'data/cache/hume_AI_analysis')
    if directory_settings.get('use_relative_paths', True):
        script_dir = Path(__file__).parent.parent.parent
        hume_analysis_dir = script_dir / hume_base_path
    else:
        hume_analysis_dir = Path(hume_base_path)
    
    if not hume_analysis_dir.exists():
        LOGGER.error(f"Hume analysis directory not found: {hume_analysis_dir}")
        return {'success': False, 'error': f'Directory not found: {hume_analysis_dir}'}
    
    LOGGER.info(f"Using hume analysis directory: {hume_analysis_dir}")
    
    # Process batch configurations if enabled
    if batch_config.get('enabled', False):
        return process_batch_configurations(batch_config, hume_analysis_dir, base_config, analysis_parameters, validation_settings)
    
    # Single configuration processing
    return process_single_configuration(
        selection_criteria, analysis_parameters, hume_analysis_dir, 
        base_config, validation_settings, output_settings
    )

def process_batch_configurations(batch_config: dict, hume_analysis_dir: Path, base_config: dict, 
                                base_analysis_params: dict, validation_settings: dict) -> Dict[str, Any]:
    """Process multiple batch configurations."""
    LOGGER.info("Processing batch configurations")
    
    configurations = batch_config.get('configurations', [])
    results = {'success': True, 'batch_results': [], 'summary': {'total': 0, 'successful': 0, 'failed': 0}}
    
    for i, config in enumerate(configurations):
        config_name = config.get('name', f'config_{i+1}')
        LOGGER.info(f"\n--- Processing batch configuration: {config_name} ---")
        
        # Create selection criteria for this batch
        selection_criteria = {
            'campaigns': {'selected': config.get('campaigns', [])},
            'sales_reps': {'selected': config.get('sales_reps', [])},
            'analysis_modes': {'selected': config.get('analysis_modes', ['all'])}
        }
        
        # Merge custom parameters
        merged_params = merge_analysis_parameters(base_analysis_params, config.get('custom_parameters', {}))
        
        # Process this configuration
        batch_result = process_single_configuration(
            selection_criteria, merged_params, hume_analysis_dir, 
            base_config, validation_settings, {}, config_name
        )
        
        results['batch_results'].append({
            'name': config_name,
            'result': batch_result
        })
        
        # Update summary
        results['summary']['total'] += 1
        if batch_result['success']:
            results['summary']['successful'] += 1
        else:
            results['summary']['failed'] += 1
    
    LOGGER.info(f"\nBatch processing complete: {results['summary']['successful']}/{results['summary']['total']} successful")
    return results

def process_single_configuration(selection_criteria: dict, analysis_parameters: dict, 
                               hume_analysis_dir: Path, base_config: dict, 
                               validation_settings: dict, output_settings: dict = None, 
                               config_name: str = None) -> Dict[str, Any]:
    """Process a single configuration (either standalone or part of batch)."""
    
    config_name = config_name or "single_config"
    LOGGER.info(f"Processing configuration: {config_name}")
    
    # Get analysis configuration
    base_analysis_config = get_analysis_config(base_config)
    
    # Merge with custom analysis parameters
    analysis_config = merge_analysis_parameters(base_analysis_config, analysis_parameters)
    analysis_config.update(validation_settings.get('validation_rules', {}))
    
    # Extract specific thresholds from analysis_parameters
    if 'hesitancy_analysis' in analysis_parameters:
        hesitancy_config = analysis_parameters['hesitancy_analysis']
        analysis_config['hesitancy_score_threshold'] = hesitancy_config.get('score_threshold', analysis_config['score_threshold'])
        analysis_config['hesitancy_vocables'] = hesitancy_config.get('vocables', analysis_config.get('hesitancy_vocables', []))
    
    if 'emotion_analysis' in analysis_parameters:
        emotion_config = analysis_parameters['emotion_analysis']
        analysis_config['score_threshold'] = emotion_config.get('score_threshold', analysis_config['score_threshold'])
        analysis_config['peak_emotion_threshold'] = emotion_config.get('peak_emotion_threshold', analysis_config['peak_emotion_threshold'])
        analysis_config['top_n_emotions'] = emotion_config.get('top_n_emotions', analysis_config['top_n_emotions'])
    
    if 'negative_emotion_analysis' in analysis_parameters:
        negative_config = analysis_parameters['negative_emotion_analysis']
        analysis_config['negative_emotions'] = negative_config.get('negative_emotions', analysis_config['negative_emotions'])
    
    # Resolve target sales reps
    campaigns = selection_criteria.get('campaigns', {}).get('selected', [])
    sales_reps = selection_criteria.get('sales_reps', {}).get('selected', [])
    analysis_modes = selection_criteria.get('analysis_modes', {}).get('selected', ['all'])
    
    # Sales reps take precedence over campaigns
    if sales_reps:
        LOGGER.info(f"Using sales rep selection: {sales_reps}")
        # Get all campaign directories to search for sales reps
        campaign_dirs = resolve_campaigns(hume_analysis_dir, ['all'])
        target_sales_rep_dirs = resolve_sales_reps(campaign_dirs, sales_reps)
    else:
        LOGGER.info(f"Using campaign selection: {campaigns}")
        campaign_dirs = resolve_campaigns(hume_analysis_dir, campaigns)
        target_sales_rep_dirs = resolve_sales_reps(campaign_dirs, ['all'])
    
    if not target_sales_rep_dirs:
        LOGGER.error("No valid sales reps found for processing")
        return {'success': False, 'error': 'No valid sales reps found'}
    
    # Process each sales rep
    results = {'success': True, 'processed_reps': [], 'summary': {'total': 0, 'successful': 0, 'failed': 0}}
    
    for sales_rep_dir in target_sales_rep_dirs:
        result = process_sales_rep(sales_rep_dir, analysis_config, analysis_modes)
        results['processed_reps'].append(result)
        results['summary']['total'] += 1
        
        if result['success']:
            results['summary']['successful'] += 1
        else:
            results['summary']['failed'] += 1
            # Handle error based on settings
            error_action = validation_settings.get('missing_sales_rep_action', 'warn_and_skip')
            if error_action == 'fail':
                return {'success': False, 'error': f"Failed processing {sales_rep_dir.name}: {result['error']}"}
    
    # Save processing summary if requested
    if output_settings and output_settings.get('save_processing_summary', False):
        save_processing_summary(results, hume_analysis_dir, output_settings, config_name)
    
    LOGGER.info(f"\nConfiguration processing complete:")
    LOGGER.info(f"  - Total sales reps processed: {results['summary']['total']}")
    LOGGER.info(f"  - Successful career analyses: {results['summary']['successful']}")
    LOGGER.info(f"  - Failed: {results['summary']['failed']}")
    
    return results

def save_processing_summary(results: dict, hume_analysis_dir: Path, output_settings: dict, config_name: str):
    """Save processing summary to file."""
    try:
        summary_file = output_settings.get('summary_file_name', 'career_analysis_processing_summary.json')
        summary_path = hume_analysis_dir / f"{config_name}_{summary_file}"
        
        summary_data = {
            'processing_date': datetime.now().isoformat(),
            'configuration_name': config_name,
            'results': results
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        LOGGER.info(f"Processing summary saved to: {summary_path}")
    except Exception as e:
        LOGGER.error(f"Failed to save processing summary: {e}")

def process_all_sales_reps(hume_analysis_dir: Optional[Path] = None, config_path: str = "config/analysis_config/audio_analysis.json") -> None:
    """Legacy function: Process all sales reps in the hume_AI_analysis directory."""
    
    # Find hume analysis directory if not provided
    if hume_analysis_dir is None:
        hume_analysis_dir = find_hume_analysis_directory()
    
    if hume_analysis_dir is None:
        LOGGER.error("Could not find hume_AI_analysis directory")
        sys.exit(1)
    
    LOGGER.info(f"Using hume analysis directory: {hume_analysis_dir}")
    #hi
    # Load configuration
    config = load_config(config_path)
    analysis_config = get_analysis_config(config)
    
    LOGGER.info(f"Configuration loaded:")
    LOGGER.info(f"  - Score threshold: {analysis_config['score_threshold']}")
    LOGGER.info(f"  - Peak emotion threshold: {analysis_config['peak_emotion_threshold']}")
    LOGGER.info(f"  - Top N emotions: {analysis_config['top_n_emotions']}")
    
    # Find all campaign rep directories
    campaign_dirs = [d for d in hume_analysis_dir.iterdir() if d.is_dir() and d.name.endswith('_reps')]
    
    if not campaign_dirs:
        LOGGER.warning("No campaign rep directories found (directories ending with '_reps')")
        return
    
    LOGGER.info(f"Found {len(campaign_dirs)} campaign directories")
    
    total_reps_processed = 0
    total_reps_successful = 0
    
    for campaign_dir in campaign_dirs:
        LOGGER.info(f"Processing campaign: {campaign_dir.name}")
        
        # Find all sales rep directories
        sales_rep_dirs = [d for d in campaign_dir.iterdir() if d.is_dir()]
        
        LOGGER.info(f"  Found {len(sales_rep_dirs)} sales reps")
        
        for sales_rep_dir in sales_rep_dirs:
            total_reps_processed += 1
            result = process_sales_rep(sales_rep_dir, analysis_config)
            if result['success']:
                total_reps_successful += 1
    
    LOGGER.info(f"\nProcessing complete:")
    LOGGER.info(f"  - Total sales reps processed: {total_reps_processed}")
    LOGGER.info(f"  - Successful career analyses: {total_reps_successful}")
    LOGGER.info(f"  - Failed: {total_reps_processed - total_reps_successful}")

def process_single_sales_rep(sales_rep_path: str, config_path: str = "config/analysis_config/audio_analysis.json") -> None:
    """Legacy function: Process a single sales rep directory."""
    sales_rep_dir = Path(sales_rep_path)
    
    if not sales_rep_dir.exists() or not sales_rep_dir.is_dir():
        LOGGER.error(f"Sales rep directory not found: {sales_rep_path}")
        sys.exit(1)
    
    # Load configuration
    config = load_config(config_path)
    analysis_config = get_analysis_config(config)
    
    # Process the single sales rep
    result = process_sales_rep(sales_rep_dir, analysis_config)
    
    if result['success']:
        LOGGER.info("Career analysis completed successfully")
    else:
        LOGGER.error(f"Career analysis failed: {result['error']}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Generate career aggregated analysis for sales reps from Hume AI analysis data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use career configuration (recommended)
  python career_aggregated_analysis.py --career-config
  
  # Use custom career config file
  python career_aggregated_analysis.py --career-config config/my_career_config.json
  
  # Legacy mode: process all sales reps
  python career_aggregated_analysis.py --hume-dir data/cache/hume_AI_analysis
  
  # Legacy mode: process single sales rep
  python career_aggregated_analysis.py --sales-rep path/to/sales/rep
        """
    )
    
    # New career configuration mode
    parser.add_argument(
        "--career-config",
        nargs='?',
        const='default',
        type=str,
        help="Use career configuration file. If no path provided, uses default location."
    )
    
    # Legacy arguments for backwards compatibility
    parser.add_argument(
        "--hume-dir",
        type=str,
        help="[Legacy] Path to hume_AI_analysis directory (auto-detected if not provided)"
    )
    parser.add_argument(
        "--sales-rep",
        type=str,
        help="[Legacy] Process a single sales rep directory instead of all"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/analysis_config/audio_analysis.json",
        help="Path to base audio_analysis.json file"
    )
    
    args = parser.parse_args()
    
    # Priority: career-config > sales-rep > hume-dir
    if args.career_config is not None:
        # Use new career configuration mode
        LOGGER.info("Using career configuration mode")
        career_config_path = None if args.career_config == 'default' else args.career_config
        
        result = process_with_career_config(career_config_path, args.config)
        
        if not result['success']:
            LOGGER.error(f"Career analysis failed: {result['error']}")
            sys.exit(1)
        else:
            LOGGER.info("Career analysis completed successfully")
            
    elif args.sales_rep:
        # Legacy mode: process single sales rep
        LOGGER.info("Using legacy single sales rep mode")
        process_single_sales_rep(args.sales_rep, args.config)
        
    else:
        # Legacy mode: process all sales reps
        LOGGER.info("Using legacy batch processing mode")
        hume_dir = Path(args.hume_dir) if args.hume_dir else None
        process_all_sales_reps(hume_dir, args.config)

if __name__ == "__main__":
    main()