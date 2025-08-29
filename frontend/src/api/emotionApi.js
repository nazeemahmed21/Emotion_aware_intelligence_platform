// API integration for Emotion Analysis Backend
// This file handles communication with your Python Streamlit backend

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8501';

/**
 * Analyzes audio file using the Hume AI backend
 * @param {Blob} audioBlob - The recorded audio blob
 * @returns {Promise<Object>} Analysis results
 */
export const analyzeEmotion = async (audioBlob) => {
  try {
    // Create FormData for file upload
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    
    // TODO: Replace this endpoint with your actual backend API
    // Option 1: Direct integration with your Python backend
    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      body: formData,
      headers: {
        // Don't set Content-Type header - let browser set it with boundary
      },
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const results = await response.json();
    
    // Transform the results to match our frontend expectations
    return transformHumeResults(results);
    
  } catch (error) {
    console.error('Error analyzing emotion:', error);
    throw new Error(`Failed to analyze audio: ${error.message}`);
  }
};

/**
 * Transform Hume AI results to frontend format
 * @param {Object} humeResults - Raw results from Hume AI
 * @returns {Object} Transformed results for frontend
 */
const transformHumeResults = (humeResults) => {
  // Extract emotions from Hume AI response structure
  const emotions = extractEmotionsFromHume(humeResults);
  
  // Calculate summary statistics
  const summary = calculateSummary(emotions);
  
  // Generate AI response based on emotions
  const aiResponse = generateAIResponse(emotions, summary);
  
  return {
    emotions,
    summary,
    aiResponse,
    rawData: humeResults // Include raw data for debugging
  };
};

/**
 * Extract emotions from Hume AI prediction structure
 * This matches the structure from your Python backend
 */
const extractEmotionsFromHume = (predictions) => {
  const allEmotions = [];
  
  try {
    // Handle the prediction structure from your Python backend
    const predictionsList = Array.isArray(predictions) ? predictions : [predictions];
    
    for (const filePrediction of predictionsList) {
      const results = filePrediction.results || filePrediction;
      const predictions = results.predictions || [results];
      
      for (const pred of predictions) {
        const models = pred.models || {};
        
        // Process each model (prosody, burst, language)
        for (const [modelName, modelData] of Object.entries(models)) {
          const groupedPredictions = modelData.grouped_predictions || [];
          
          for (const group of groupedPredictions) {
            const predictions = group.predictions || [];
            
            for (const prediction of predictions) {
              const emotions = prediction.emotions || [];
              
              for (const emotion of emotions) {
                allEmotions.push({
                  name: emotion.name,
                  score: emotion.score,
                  model: modelName,
                  timestamp: prediction.time || {}
                });
              }
            }
          }
        }
      }
    }
    
    return allEmotions;
  } catch (error) {
    console.error('Error extracting emotions:', error);
    return [];
  }
};

/**
 * Calculate summary statistics from emotions
 */
const calculateSummary = (emotions) => {
  if (!emotions.length) return null;
  
  // Group emotions by name
  const emotionStats = {};
  emotions.forEach(emotion => {
    if (!emotionStats[emotion.name]) {
      emotionStats[emotion.name] = {
        scores: [],
        models: [],
        totalScore: 0,
        count: 0
      };
    }
    
    emotionStats[emotion.name].scores.push(emotion.score);
    emotionStats[emotion.name].models.push(emotion.model);
    emotionStats[emotion.name].totalScore += emotion.score;
    emotionStats[emotion.name].count += 1;
  });
  
  // Calculate final stats
  Object.keys(emotionStats).forEach(name => {
    const stats = emotionStats[name];
    stats.meanScore = stats.totalScore / stats.count;
    stats.maxScore = Math.max(...stats.scores);
    stats.uniqueModels = [...new Set(stats.models)];
  });
  
  // Model breakdown
  const modelBreakdown = {
    prosody: { emotionCount: 0, avgScore: 0, totalScore: 0 },
    burst: { emotionCount: 0, avgScore: 0, totalScore: 0 },
    language: { emotionCount: 0, avgScore: 0, totalScore: 0 }
  };
  
  emotions.forEach(emotion => {
    if (modelBreakdown[emotion.model]) {
      modelBreakdown[emotion.model].emotionCount++;
      modelBreakdown[emotion.model].totalScore += emotion.score;
    }
  });
  
  // Calculate averages
  Object.keys(modelBreakdown).forEach(model => {
    const data = modelBreakdown[model];
    data.avgScore = data.emotionCount > 0 ? data.totalScore / data.emotionCount : 0;
  });
  
  // Hesitancy analysis
  const hesitancyKeywords = [
    'anxiety', 'nervousness', 'uncertainty', 'confusion',
    'hesitation', 'doubt', 'worry', 'stress', 'awkwardness'
  ];
  
  const hesitancyEmotions = Object.entries(emotionStats)
    .filter(([name]) => hesitancyKeywords.some(keyword => 
      name.toLowerCase().includes(keyword)
    ))
    .map(([name, stats]) => ({ name, score: stats.meanScore }));
  
  const hesitancyAnalysis = {
    indicators: hesitancyEmotions.length,
    averageScore: hesitancyEmotions.length > 0 
      ? hesitancyEmotions.reduce((sum, e) => sum + e.score, 0) / hesitancyEmotions.length 
      : 0,
    level: hesitancyEmotions.length > 0 
      ? (hesitancyEmotions.reduce((sum, e) => sum + e.score, 0) / hesitancyEmotions.length > 0.5 ? 'Low' : 'High')
      : 'High',
    patterns: hesitancyEmotions.map(e => e.name)
  };
  
  return {
    totalEmotions: Object.keys(emotionStats).length,
    dominantEmotion: Object.entries(emotionStats)
      .sort(([,a], [,b]) => b.meanScore - a.meanScore)[0],
    averageIntensity: emotions.reduce((sum, e) => sum + e.score, 0) / emotions.length,
    modelBreakdown,
    hesitancyAnalysis
  };
};

/**
 * Generate AI response based on emotion analysis
 */
const generateAIResponse = (emotions, summary) => {
  if (!emotions.length || !summary) {
    return "I wasn't able to detect clear emotional patterns in this recording. This might be due to audio quality or neutral speech content.";
  }
  
  const [dominantName, dominantStats] = summary.dominantEmotion;
  const intensity = summary.averageIntensity;
  
  let response = `Based on my analysis of your voice, I detected ${summary.totalEmotions} distinct emotions. `;
  
  response += `The most prominent emotion was **${dominantName}** with a confidence of ${(dominantStats.meanScore * 100).toFixed(1)}%. `;
  
  if (intensity > 0.6) {
    response += "Your speech shows high emotional intensity, suggesting strong feelings or engagement with the topic. ";
  } else if (intensity > 0.3) {
    response += "Your emotional expression appears moderate and balanced. ";
  } else {
    response += "Your speech shows relatively calm and controlled emotional expression. ";
  }
  
  if (summary.hesitancyAnalysis.indicators > 0) {
    response += `I also noticed ${summary.hesitancyAnalysis.indicators} indicators of hesitancy or uncertainty, which might suggest some internal processing or careful consideration of your words.`;
  } else {
    response += "Your speech appears confident and decisive without significant hesitancy patterns.";
  }
  
  return response;
};

// Mock data for development/testing
export const getMockEmotionResults = () => ({
  emotions: [
    { name: 'joy', score: 0.85, model: 'prosody' },
    { name: 'excitement', score: 0.72, model: 'burst' },
    { name: 'confidence', score: 0.68, model: 'language' },
    { name: 'curiosity', score: 0.45, model: 'prosody' },
    { name: 'calmness', score: 0.38, model: 'language' },
    { name: 'surprise', score: 0.32, model: 'burst' },
    { name: 'interest', score: 0.28, model: 'prosody' },
    { name: 'contentment', score: 0.25, model: 'language' },
    { name: 'anticipation', score: 0.22, model: 'burst' },
    { name: 'satisfaction', score: 0.18, model: 'prosody' }
  ],
  summary: {
    totalEmotions: 10,
    dominantEmotion: ['joy', { meanScore: 0.85, count: 1 }],
    averageIntensity: 0.433,
    modelBreakdown: {
      prosody: { emotionCount: 4, avgScore: 0.445 },
      burst: { emotionCount: 3, avgScore: 0.42 },
      language: { emotionCount: 3, avgScore: 0.437 }
    },
    hesitancyAnalysis: {
      indicators: 0,
      averageScore: 0,
      level: 'High',
      patterns: []
    }
  },
  aiResponse: "Based on my analysis of your voice, I detected 10 distinct emotions. The most prominent emotion was **joy** with a confidence of 85.0%. Your speech shows moderate emotional intensity and appears confident without significant hesitancy patterns."
});