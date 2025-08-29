import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, MicOff, Square, Play, Pause } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { cn, formatTime } from '../lib/utils';
import WaveformVisualizer from './WaveformVisualizer';
import EmotionResults from './EmotionResults';
import { analyzeEmotion, getMockEmotionResults } from '../api/emotionApi';

const VoiceRecorder = () => {
  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  
  // Analysis state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [emotionResults, setEmotionResults] = useState(null);
  const [error, setError] = useState(null);
  
  // Refs
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const streamRef = useRef(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  // Timer effect
  useEffect(() => {
    if (isRecording && !isPaused) {
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [isRecording, isPaused]);

  const startRecording = async () => {
    try {
      setError(null);
      
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100
        } 
      });
      
      streamRef.current = stream;
      
      // Create MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { 
          type: 'audio/webm;codecs=opus' 
        });
        setAudioBlob(audioBlob);
        setAudioUrl(URL.createObjectURL(audioBlob));
        
        // Clean up stream
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
      };
      
      // Start recording
      mediaRecorder.start(100); // Collect data every 100ms
      setIsRecording(true);
      setRecordingTime(0);
      
    } catch (err) {
      console.error('Error starting recording:', err);
      setError('Failed to access microphone. Please check permissions.');
    }
  };

  const pauseRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      if (isPaused) {
        mediaRecorderRef.current.resume();
        setIsPaused(false);
      } else {
        mediaRecorderRef.current.pause();
        setIsPaused(true);
      }
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsPaused(false);
    }
  };

  const analyzeAudio = async () => {
    if (!audioBlob) return;
    
    setIsAnalyzing(true);
    setAnalysisProgress(0);
    setError(null);
    
    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setAnalysisProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 500);
      
      // Call the emotion analysis API
      // TODO: Replace getMockEmotionResults() with analyzeEmotion(audioBlob) 
      // when your backend API is ready
      let results;
      
      if (process.env.NODE_ENV === 'development') {
        // Use mock data in development
        await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate API delay
        results = getMockEmotionResults();
      } else {
        // Use real API in production
        results = await analyzeEmotion(audioBlob);
      }
      
      clearInterval(progressInterval);
      setAnalysisProgress(100);
      setEmotionResults(results);
      
    } catch (err) {
      console.error('Analysis error:', err);
      setError(`Failed to analyze audio: ${err.message}`);
    } finally {
      setIsAnalyzing(false);
      setTimeout(() => setAnalysisProgress(0), 1000);
    }
  };

  const resetRecording = () => {
    setAudioBlob(null);
    setAudioUrl(null);
    setRecordingTime(0);
    setEmotionResults(null);
    setError(null);
    audioChunksRef.current = [];
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 space-y-8">
      {/* Header */}
      <div className="text-center space-y-4">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            🧠 Emotion-Aware Voice Analyzer
          </h1>
          <p className="text-lg text-muted-foreground mt-2">
            Record your voice and discover the emotions within
          </p>
        </motion.div>
      </div>

      {/* Main Recording Interface */}
      <Card className="relative overflow-hidden">
        <CardHeader className="text-center pb-8">
          <CardTitle className="text-2xl">Voice Recording Studio</CardTitle>
        </CardHeader>
        
        <CardContent className="space-y-8">
          {/* Recording Controls */}
          <div className="flex flex-col items-center space-y-6">
            
            {/* Main Record Button */}
            <div className="relative">
              <motion.div
                className={cn(
                  "absolute inset-0 rounded-full",
                  isRecording && !isPaused && "animate-pulse-ring bg-red-400"
                )}
              />
              
              <Button
                onClick={isRecording ? stopRecording : startRecording}
                size="xl"
                variant={isRecording ? "destructive" : "gradient"}
                className={cn(
                  "relative w-20 h-20 rounded-full shadow-2xl transition-all duration-300",
                  "hover:scale-105 active:scale-95",
                  isRecording && "bg-red-500 hover:bg-red-600"
                )}
                disabled={isAnalyzing}
                aria-label={isRecording ? "Stop recording" : "Start recording"}
              >
                <motion.div
                  key={isRecording ? "recording" : "idle"}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: "spring", stiffness: 300 }}
                >
                  {isRecording ? (
                    <Square className="w-8 h-8" />
                  ) : (
                    <Mic className="w-8 h-8" />
                  )}
                </motion.div>
              </Button>
            </div>

            {/* Recording Status */}
            <AnimatePresence>
              {isRecording && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="text-center space-y-4"
                >
                  <div className="flex items-center justify-center space-x-4">
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
                      <span className="text-lg font-medium">
                        {isPaused ? 'Paused' : 'Recording'}
                      </span>
                    </div>
                    
                    <div className="text-2xl font-mono font-bold text-primary">
                      {formatTime(recordingTime)}
                    </div>
                  </div>

                  {/* Pause/Resume Button */}
                  <Button
                    onClick={pauseRecording}
                    variant="outline"
                    size="sm"
                    className="min-w-24"
                  >
                    {isPaused ? (
                      <>
                        <Play className="w-4 h-4 mr-2" />
                        Resume
                      </>
                    ) : (
                      <>
                        <Pause className="w-4 h-4 mr-2" />
                        Pause
                      </>
                    )}
                  </Button>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Waveform Visualizer */}
            <div className="w-full max-w-md">
              <WaveformVisualizer 
                isRecording={isRecording && !isPaused}
                stream={streamRef.current}
              />
            </div>
          </div>

          {/* Audio Playback */}
          <AnimatePresence>
            {audioUrl && !isRecording && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="space-y-4"
              >
                <div className="text-center">
                  <h3 className="text-lg font-semibold mb-4">Recording Complete</h3>
                  <audio 
                    controls 
                    src={audioUrl} 
                    className="w-full max-w-md mx-auto"
                    preload="metadata"
                  />
                </div>

                {/* Action Buttons */}
                <div className="flex justify-center space-x-4">
                  <Button
                    onClick={analyzeAudio}
                    disabled={isAnalyzing}
                    variant="gradient"
                    size="lg"
                    className="min-w-32"
                  >
                    {isAnalyzing ? 'Analyzing...' : 'Analyze Emotions'}
                  </Button>
                  
                  <Button
                    onClick={resetRecording}
                    variant="outline"
                    size="lg"
                  >
                    Record Again
                  </Button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Analysis Progress */}
          <AnimatePresence>
            {isAnalyzing && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-4"
              >
                <div className="text-center">
                  <h3 className="text-lg font-semibold mb-2">
                    Analyzing Emotions with AI
                  </h3>
                  <p className="text-muted-foreground">
                    Processing your voice with Hume AI models...
                  </p>
                </div>
                
                <div className="space-y-2">
                  <Progress value={analysisProgress} className="h-2" />
                  <div className="text-center text-sm text-muted-foreground">
                    {analysisProgress}% Complete
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Error Display */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="bg-red-50 border border-red-200 rounded-lg p-4 text-center"
              >
                <p className="text-red-700 font-medium">{error}</p>
                <Button
                  onClick={() => setError(null)}
                  variant="ghost"
                  size="sm"
                  className="mt-2 text-red-600 hover:text-red-700"
                >
                  Dismiss
                </Button>
              </motion.div>
            )}
          </AnimatePresence>
        </CardContent>
      </Card>

      {/* Emotion Analysis Results */}
      <AnimatePresence>
        {emotionResults && (
          <EmotionResults results={emotionResults} />
        )}
      </AnimatePresence>

      {/* Instructions */}
      <Card className="bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
        <CardContent className="pt-6">
          <div className="grid md:grid-cols-3 gap-6 text-center">
            <div className="space-y-2">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto">
                <Mic className="w-6 h-6 text-blue-600" />
              </div>
              <h3 className="font-semibold">1. Record</h3>
              <p className="text-sm text-muted-foreground">
                Click the microphone to start recording your voice
              </p>
            </div>
            
            <div className="space-y-2">
              <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto">
                <Square className="w-6 h-6 text-purple-600" />
              </div>
              <h3 className="font-semibold">2. Stop</h3>
              <p className="text-sm text-muted-foreground">
                Click stop when finished, then review your recording
              </p>
            </div>
            
            <div className="space-y-2">
              <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto">
                <span className="text-green-600 font-bold">🧠</span>
              </div>
              <h3 className="font-semibold">3. Analyze</h3>
              <p className="text-sm text-muted-foreground">
                Get AI-powered insights into your emotional expressions
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default VoiceRecorder;