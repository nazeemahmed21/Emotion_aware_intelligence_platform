# Hume AI Integration

This module provides comprehensive integration with Hume AI's emotion analysis API, including client libraries, config-driven analysis, and career aggregation capabilities.

## Features

- **Config-Driven Analysis**: Process audio files through configurable analysis workflows
- **Career Aggregation**: Analyze emotion patterns across multiple sales representatives and campaigns
- **Multiple Granularity Levels**: Support for word, sentence, utterance, and turn-level analysis
- **Batch Processing**: Process multiple files and campaigns efficiently  
- **Flexible Configuration**: JSON-based configuration for analysis parameters and selection

## Files Overview

### Core Components

- **`hume_client.py`**: Main client for interacting with Hume AI API
- **`config_driven_analysis.py`**: Config-driven analysis system for flexible audio processing
- **`career_aggregated_analysis.py`**: Career-level analysis across multiple sales reps

### Configuration & Core Utilities

- **`core/config_loader.py`**: Configuration loading with caching and validation
- **`core/analysis_config.py`**: Analysis-specific configuration extraction
- **`core/emotion_processor.py`**: Emotion processing and statistical analysis
- **`core/prediction_parser.py`**: Hume AI prediction file parsing utilities

### Configuration Scripts

- **`config_based_analysis_scripts/run_full_conversation_analysis.py`**: Full conversation analysis runner
- **`config_based_analysis_scripts/run_segment_analysis.py`**: Individual segment analysis runner

### Testing & Utilities

- **`test_hume_integration.py`**: Test suite for verifying integration
- **`test_config_driven_analysis.py`**: Test suite for config-driven analysis
- **`__init__.py`**: Package initialization and exports

## Configuration

All settings are configured through `config/analysis_config/audio_analysis.json`:

```json
{
  "hume_ai": {
    "api_key": "",
    "secret_key": "",
    "webhook_url": "",
    "default_granularity": "utterance",
    "score_threshold": 0.6,
    "top_n_emotions": 10,
    "batch_size": 100,
    "negative_emotions": [...],
    "hesitancy_vocables": [...],
    "intensity_thresholds": {...},
    "hesitancy_thresholds": {...}
  }
}
```

## Environment Variables

Set these in your `.env` file:

```bash
HUME_API_KEY=your_api_key_here
HUME_SECRET_KEY=your_secret_key_here  # Optional, for OAuth2
HUME_WEBHOOK_URL=your_webhook_url     # Optional
```

## Usage Examples

### 1. Test Integration

```bash
python src/hume/test_hume_integration.py
```

### 2. Test Config-Driven Analysis Setup

```bash
python src/hume/test_config_driven_analysis.py
```

### 3. Run Config-Driven Analysis

```bash
python src/hume/config_driven_analysis.py
```

### 4. Run Career Aggregated Analysis

```bash
python src/hume/career_aggregated_analysis.py
```

### 5. Run Specific Analysis Scripts

```bash
# Full conversation analysis
python src/hume/config_based_analysis_scripts/run_full_conversation_analysis.py

# Individual segment analysis  
python src/hume/config_based_analysis_scripts/run_segment_analysis.py
```

## Granularity Levels

- **WORD**: Analyze emotions at the word level
- **SENTENCE**: Analyze emotions at the sentence level  
- **UTTERANCE**: Analyze emotions at the utterance level (default)
- **TURN**: Analyze emotions at the conversational turn level (includes speaker identification)

## Output Files

### Emotion Analysis Results

- `{filename}_{timestamp}_predictions.json`: Raw Hume AI predictions
- `{filename}_{timestamp}_artifacts.zip`: Additional artifacts (CSV, etc.)
- `{job_name}_combined_results.json`: Combined results for batch processing

### Analysis Reports

The analysis tools generate comprehensive reports including:

- **Emotion Summary**: Average scores for all detected emotions
- **Peak Moments**: Highest scoring emotional events
- **Negative Emotion Analysis**: Analysis of negative emotional patterns
- **Hesitancy Analysis**: Detection and analysis of hesitancy vocables
- **Per-File Hesitancy Breakdown**: Individual hesitancy scores for each audio file
- **Statistical Analysis**: Outlier detection and statistical measures

### Career Analysis Output Format

The career aggregated analysis now includes detailed per-file hesitancy metrics:

```json
{
  "hesitancy_vocable_analysis": {
    "total_hesitancy_events": 15,
    "average_hesitancy_per_audio_file": 2.5,
    "per_file_hesitancy_analysis": {
      "summary_average_hesitancy_per_file": {
        "contact_123_analysis_results.json": 0.81,
        "contact_456_analysis_results.json": 0.73,
        "contact_789_analysis_results.json": 0.65
      },
      "detailed_per_file_metrics": {
        "contact_123_analysis_results.json": {
          "total_hesitancy_events": 8,
          "average_hesitancy_score": 0.81,
          "hesitancy_vocables_detected": {"Uh": 3, "Umm": 2, "Hmm": 3}
        }
      }
    }
  }
}
```

## API Integration

### HumeClient Class

```python
from hume import HumeClient, HumeConfig, GranularityLevel

# Initialize client
config = HumeConfig.from_config_file()
client = HumeClient(config)

# Submit job
job_id = await client.submit_files(['audio.wav'], GranularityLevel.UTTERANCE)

# Wait for completion
success = await client.wait_for_job(job_id)

# Get results
predictions = await client.get_job_predictions(job_id)
```

### ConfigDrivenHumeAnalyzer Class

```python
from hume import ConfigDrivenHumeAnalyzer

# Initialize analyzer
analyzer = ConfigDrivenHumeAnalyzer()

# Process configured analysis
results = await analyzer.process_analysis_selection()

# Get career analysis
career_analyzer = analyzer.career_aggregated_analysis
career_results = career_analyzer.analyze_career_data()
```

## Statistical Analysis

The module includes advanced statistical analysis capabilities:

- **Threshold Analysis**: Filter emotions by configurable score thresholds
- **Outlier Detection**: Identify statistically significant emotional events
- **Intensity Classification**: Classify emotions by intensity levels
- **Hesitancy Detection**: Analyze speech hesitancy patterns with per-file breakdowns
- **Per-File Analysis**: Individual hesitancy metrics for each audio file analyzed

## Error Handling

The integration includes comprehensive error handling:

- **API Errors**: Proper handling of Hume AI API errors
- **Network Issues**: Retry logic with configurable parameters
- **File Errors**: Graceful handling of missing or invalid files
- **Configuration Errors**: Clear error messages for configuration issues

## Logging

All components use structured logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Log levels can be configured in `config.json`:

```json
{
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s"
  }
}
```

## Troubleshooting

### Common Issues

1. **API Key Not Set**: Ensure `HUME_API_KEY` is set in your `.env` file
2. **Config File Not Found**: Ensure `config.json` exists in the project root
3. **Network Errors**: Check internet connection and API endpoint availability
4. **File Format Issues**: Ensure audio files are in supported formats (WAV, MP3, M4A, OGG, MP4)

### Debug Mode

Enable debug logging for detailed information:

```bash
export PYTHONPATH=src
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from hume.test_hume_integration import main
main()
"
```

## Dependencies

- `requests`: HTTP client for API communication
- `boto3`: AWS S3 integration
- `pandas`: Data analysis and manipulation
- `numpy`: Numerical computing
- `pathlib`: Path handling
- `asyncio`: Asynchronous programming

## License

This module is part of the emotion analysis workflow project and follows the same licensing terms.