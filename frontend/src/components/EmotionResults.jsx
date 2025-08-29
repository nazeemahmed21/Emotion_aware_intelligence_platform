import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, Brain, Heart, Zap } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { formatEmotionScore, getEmotionIntensity, getEmotionColor } from '../lib/utils';

const EmotionResults = ({ results }) => {
  if (!results || !results.emotions) {
    return null;
  }

  const { emotions, summary, aiResponse } = results;
  
  // Sort emotions by score
  const topEmotions = emotions
    .sort((a, b) => b.score - a.score)
    .slice(0, 10);

  const dominantEmotion = topEmotions[0];
  const averageIntensity = emotions.reduce((sum, e) => sum + e.score, 0) / emotions.length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="space-y-6"
    >
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200">
          <CardContent className="p-4 text-center">
            <Brain className="w-8 h-8 text-blue-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-blue-700">
              {emotions.length}
            </div>
            <div className="text-sm text-blue-600">Emotions Detected</div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200">
          <CardContent className="p-4 text-center">
            <Heart className="w-8 h-8 text-purple-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-purple-700 capitalize">
              {dominantEmotion?.name || 'N/A'}
            </div>
            <div className="text-sm text-purple-600">Dominant Emotion</div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-green-50 to-green-100 border-green-200">
          <CardContent className="p-4 text-center">
            <TrendingUp className="w-8 h-8 text-green-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-green-700">
              {formatEmotionScore(averageIntensity)}%
            </div>
            <div className="text-sm text-green-600">Avg Intensity</div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-orange-50 to-orange-100 border-orange-200">
          <CardContent className="p-4 text-center">
            <Zap className="w-8 h-8 text-orange-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-orange-700">
              {formatEmotionScore(dominantEmotion?.score || 0)}%
            </div>
            <div className="text-sm text-orange-600">Peak Score</div>
          </CardContent>
        </Card>
      </div>

      {/* Top Emotions Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <TrendingUp className="w-5 h-5" />
            <span>Top 10 Emotions Detected</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {topEmotions.map((emotion, index) => {
            const intensity = getEmotionIntensity(emotion.score);
            const colorClass = getEmotionColor(emotion.name);
            
            return (
              <motion.div
                key={emotion.name}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="space-y-2"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="text-lg font-medium">#{index + 1}</div>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${colorClass}`}>
                      {emotion.name}
                    </span>
                    <span className={`text-sm font-medium ${intensity.color}`}>
                      {intensity.level}
                    </span>
                  </div>
                  <div className="text-lg font-bold">
                    {formatEmotionScore(emotion.score)}%
                  </div>
                </div>
                <Progress 
                  value={emotion.score * 100} 
                  className="h-2"
                />
              </motion.div>
            );
          })}
        </CardContent>
      </Card>

      {/* AI Analysis Response */}
      {aiResponse && (
        <Card className="bg-gradient-to-r from-indigo-50 to-purple-50 border-indigo-200">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-indigo-700">
              <Brain className="w-5 h-5" />
              <span>AI Emotional Analysis</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="prose prose-indigo max-w-none">
              <p className="text-gray-700 leading-relaxed">
                {aiResponse}
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Model Breakdown */}
      {summary?.modelBreakdown && (
        <Card>
          <CardHeader>
            <CardTitle>AI Model Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {Object.entries(summary.modelBreakdown).map(([model, data]) => {
                const modelInfo = {
                  prosody: { 
                    icon: '🎵', 
                    name: 'Prosody Model',
                    description: 'Voice tone & rhythm analysis'
                  },
                  burst: { 
                    icon: '⚡', 
                    name: 'Burst Model',
                    description: 'Quick emotional expressions'
                  },
                  language: { 
                    icon: '📝', 
                    name: 'Language Model',
                    description: 'Speech content analysis'
                  }
                };

                const info = modelInfo[model] || { icon: '🤖', name: model, description: 'AI Analysis' };

                return (
                  <Card key={model} className="text-center">
                    <CardContent className="p-4">
                      <div className="text-3xl mb-2">{info.icon}</div>
                      <h3 className="font-semibold text-lg">{info.name}</h3>
                      <p className="text-sm text-muted-foreground mb-3">
                        {info.description}
                      </p>
                      <div className="space-y-2">
                        <div className="text-2xl font-bold text-primary">
                          {data.emotionCount}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          Emotions detected
                        </div>
                        <div className="text-sm">
                          Avg Score: <span className="font-medium">
                            {formatEmotionScore(data.avgScore)}%
                          </span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Hesitancy Analysis */}
      {summary?.hesitancyAnalysis && (
        <Card className="bg-gradient-to-r from-yellow-50 to-orange-50 border-yellow-200">
          <CardHeader>
            <CardTitle className="text-yellow-700">🤔 Hesitancy & Uncertainty Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-yellow-600">
                  {formatEmotionScore(summary.hesitancyAnalysis.averageScore)}%
                </div>
                <div className="text-sm text-yellow-600">Average Hesitancy</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {summary.hesitancyAnalysis.indicators}
                </div>
                <div className="text-sm text-orange-600">Indicators Found</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600">
                  {summary.hesitancyAnalysis.level}
                </div>
                <div className="text-sm text-red-600">Confidence Level</div>
              </div>
            </div>
            
            {summary.hesitancyAnalysis.patterns?.length > 0 && (
              <div className="space-y-2">
                <h4 className="font-medium text-yellow-700">Detected Patterns:</h4>
                <div className="flex flex-wrap gap-2">
                  {summary.hesitancyAnalysis.patterns.map((pattern, index) => (
                    <span 
                      key={index}
                      className="px-3 py-1 bg-yellow-100 text-yellow-800 rounded-full text-sm"
                    >
                      {pattern}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </motion.div>
  );
};

export default EmotionResults;