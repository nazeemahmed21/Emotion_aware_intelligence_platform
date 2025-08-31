#!/usr/bin/env python3
"""
Coaching Agent for Interview Practice
====================================

This module provides intelligent coaching feedback by combining:
- User's transcribed answer
- Emotion analysis from voice
- Question context and type
- STAR method guidance
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

from config import LLM_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class CoachingContext:
    """Context for coaching analysis"""
    question_text: str
    question_category: str  # 'STAR', 'Behavioral', 'Technical'
    user_answer: str
    voice_emotions: List[Dict[str, Any]]
    transcription_confidence: float
    answer_duration: float

@dataclass
class CoachingFeedback:
    """Structured coaching feedback"""
    overall_score: float  # 0-10
    content_score: float  # 0-10
    emotion_score: float  # 0-10
    delivery_score: float  # 0-10
    
    strengths: List[str]
    areas_for_improvement: List[str]
    specific_suggestions: List[str]
    star_method_feedback: Optional[str]
    emotion_coaching: Optional[str]
    
    next_steps: List[str]
    confidence_boost: str

class CoachingAgent:
    """Intelligent coaching agent for interview practice"""
    
    def __init__(self):
        """Initialize the coaching agent"""
        self.llm_config = LLM_CONFIG
        logger.info(f"Coaching Agent initialized with {self.llm_config['provider']}")
    
    def analyze_answer(self, context: CoachingContext) -> CoachingFeedback:
        """
        Analyze user's answer and provide comprehensive coaching feedback
        
        Args:
            context: Coaching context with question, answer, and emotions
            
        Returns:
            Structured coaching feedback
        """
        try:
            logger.info("Starting comprehensive answer analysis")
            
            # Analyze different aspects
            content_analysis = self._analyze_content(context)
            emotion_analysis = self._analyze_emotions(context)
            delivery_analysis = self._analyze_delivery(context)
            
            # Generate coaching feedback
            feedback = self._generate_coaching_feedback(
                context, content_analysis, emotion_analysis, delivery_analysis
            )
            
            logger.info(f"Coaching analysis completed. Overall score: {feedback.overall_score}/10")
            return feedback
            
        except Exception as e:
            logger.error(f"Error in coaching analysis: {e}")
            return self._generate_fallback_feedback(context)
    
    def _analyze_content(self, context: CoachingContext) -> Dict[str, Any]:
        """Analyze the content quality of the answer"""
        analysis = {
            'score': 0.0,
            'strengths': [],
            'weaknesses': [],
            'star_method_usage': False,
            'specificity': 0.0,
            'relevance': 0.0
        }
        
        try:
            answer = context.user_answer.lower()
            
            # Check STAR method usage
            star_indicators = ['situation', 'task', 'action', 'result', 'outcome', 'impact']
            star_count = sum(1 for indicator in star_indicators if indicator in answer)
            analysis['star_method_usage'] = star_count >= 2
            
            # Check specificity (concrete details vs vague statements)
            concrete_words = ['because', 'specifically', 'example', 'resulted in', 'led to', 'achieved']
            vague_words = ['maybe', 'probably', 'sometimes', 'usually', 'generally']
            
            concrete_score = sum(1 for word in concrete_words if word in answer)
            vague_score = sum(1 for word in vague_words if word in answer)
            
            analysis['specificity'] = max(0, (concrete_score - vague_score) / 5.0)
            
            # Check relevance to question
            question_words = set(context.question_text.lower().split())
            answer_words = set(answer.split())
            common_words = question_words.intersection(answer_words)
            analysis['relevance'] = len(common_words) / max(len(question_words), 1)
            
            # Calculate content score
            analysis['score'] = (
                (analysis['star_method_usage'] * 3.0) +
                (analysis['specificity'] * 3.0) +
                (analysis['relevance'] * 4.0)
            ) / 10.0
            
            # Identify strengths and weaknesses
            if analysis['star_method_usage']:
                analysis['strengths'].append("Good use of STAR method")
            else:
                analysis['weaknesses'].append("Consider using STAR method for structure")
            
            if analysis['specificity'] > 0.6:
                analysis['strengths'].append("Good use of specific examples")
            else:
                analysis['weaknesses'].append("Add more concrete examples and details")
            
            if analysis['relevance'] > 0.7:
                analysis['strengths'].append("Answer is relevant to the question")
            else:
                analysis['weaknesses'].append("Ensure answer directly addresses the question")
                
        except Exception as e:
            logger.error(f"Error in content analysis: {e}")
        
        return analysis
    
    def _analyze_emotions(self, context: CoachingContext) -> Dict[str, Any]:
        """Analyze emotional aspects from voice analysis"""
        analysis = {
            'score': 0.0,
            'primary_emotion': 'neutral',
            'confidence_level': 'medium',
            'emotional_stability': 'medium',
            'recommendations': []
        }
        
        try:
            if not context.voice_emotions:
                analysis['score'] = 5.0  # Neutral score if no emotions detected
                return analysis
            
            # Find top emotions
            top_emotions = sorted(
                context.voice_emotions, 
                key=lambda x: x.get('score', 0), 
                reverse=True
            )[:3]
            
            primary_emotion = top_emotions[0] if top_emotions else {}
            analysis['primary_emotion'] = primary_emotion.get('emotion', 'neutral')
            
            # Analyze confidence indicators
            confidence_emotions = ['joy', 'excitement', 'confidence', 'determination']
            nervous_emotions = ['fear', 'anxiety', 'nervousness', 'uncertainty']
            
            if analysis['primary_emotion'] in confidence_emotions:
                analysis['confidence_level'] = 'high'
                analysis['score'] = 8.0
                analysis['recommendations'].append("Great confidence! Keep this energy")
            elif analysis['primary_emotion'] in nervous_emotions:
                analysis['confidence_level'] = 'low'
                analysis['score'] = 3.0
                analysis['recommendations'].append("Practice breathing exercises before interviews")
                analysis['recommendations'].append("Remember: you're prepared and capable")
            else:
                analysis['confidence_level'] = 'medium'
                analysis['score'] = 6.0
                analysis['recommendations'].append("Good emotional balance, could show more enthusiasm")
            
            # Check emotional stability (consistency across emotions)
            emotion_scores = [e.get('score', 0) for e in top_emotions]
            if len(emotion_scores) > 1:
                variance = sum((score - sum(emotion_scores)/len(emotion_scores))**2 for score in emotion_scores)
                if variance < 0.1:
                    analysis['emotional_stability'] = 'high'
                    analysis['score'] += 1.0
                elif variance > 0.3:
                    analysis['emotional_stability'] = 'low'
                    analysis['score'] -= 1.0
            
            analysis['score'] = max(0, min(10, analysis['score']))
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
        
        return analysis
    
    def _analyze_delivery(self, context: CoachingContext) -> Dict[str, Any]:
        """Analyze delivery aspects like pacing, clarity, etc."""
        analysis = {
            'score': 0.0,
            'pacing': 'medium',
            'clarity': 'medium',
            'engagement': 'medium',
            'recommendations': []
        }
        
        try:
            # Analyze pacing based on duration and content length
            words_per_minute = len(context.user_answer.split()) / (context.answer_duration / 60)
            
            if words_per_minute < 100:
                analysis['pacing'] = 'slow'
                analysis['recommendations'].append("Consider speaking a bit faster to maintain engagement")
            elif words_per_minute > 200:
                analysis['pacing'] = 'fast'
                analysis['recommendations'].append("Slow down slightly to ensure clarity")
            else:
                analysis['pacing'] = 'good'
                analysis['score'] += 2.0
            
            # Analyze clarity based on transcription confidence
            if context.transcription_confidence > 0.8:
                analysis['clarity'] = 'high'
                analysis['score'] += 2.0
            elif context.transcription_confidence < 0.6:
                analysis['clarity'] = 'low'
                analysis['recommendations'].append("Work on clear pronunciation and enunciation")
            else:
                analysis['clarity'] = 'medium'
                analysis['score'] += 1.0
            
            # Analyze engagement based on answer structure
            if context.user_answer.count('.') > 2:
                analysis['engagement'] = 'good'
                analysis['score'] += 1.0
            else:
                analysis['engagement'] = 'could_improve'
                analysis['recommendations'].append("Break your answer into shorter, digestible parts")
            
            # Base score
            analysis['score'] += 4.0
            analysis['score'] = max(0, min(10, analysis['score']))
            
        except Exception as e:
            logger.error(f"Error in delivery analysis: {e}")
        
        return analysis
    
    def _generate_coaching_feedback(
        self, 
        context: CoachingContext,
        content_analysis: Dict[str, Any],
        emotion_analysis: Dict[str, Any],
        delivery_analysis: Dict[str, Any]
    ) -> CoachingFeedback:
        """Generate comprehensive coaching feedback"""
        
        # Calculate overall score
        overall_score = (
            content_analysis['score'] * 0.4 +
            emotion_analysis['score'] * 0.3 +
            delivery_analysis['score'] * 0.3
        )
        
        # Generate feedback
        feedback = CoachingFeedback(
            overall_score=round(overall_score, 1),
            content_score=round(content_analysis['score'], 1),
            emotion_score=round(emotion_analysis['score'], 1),
            delivery_score=round(delivery_analysis['score'], 1),
            
            strengths=content_analysis['strengths'] + emotion_analysis['recommendations'][:1],
            areas_for_improvement=content_analysis['weaknesses'] + delivery_analysis['recommendations'],
            specific_suggestions=self._generate_specific_suggestions(context, content_analysis),
            star_method_feedback=self._generate_star_feedback(context, content_analysis),
            emotion_coaching=self._generate_emotion_coaching(emotion_analysis),
            
            next_steps=self._generate_next_steps(overall_score),
            confidence_boost=self._generate_confidence_boost(overall_score)
        )
        
        return feedback
    
    def _generate_specific_suggestions(self, context: CoachingContext, content_analysis: Dict[str, Any]) -> List[str]:
        """Generate specific, actionable suggestions"""
        suggestions = []
        
        if not content_analysis['star_method_usage']:
            suggestions.append("Structure your answer: Situation → Task → Action → Result")
        
        if content_analysis['specificity'] < 0.5:
            suggestions.append("Add specific numbers, metrics, or outcomes to your examples")
        
        if context.question_category == 'STAR':
            suggestions.append("Use the STAR method: describe a specific situation, your role, actions taken, and results achieved")
        elif context.question_category == 'Technical':
            suggestions.append("Explain your technical approach step-by-step with examples")
        
        return suggestions
    
    def _generate_star_feedback(self, context: CoachingContext, content_analysis: Dict[str, Any]) -> Optional[str]:
        """Generate STAR method specific feedback"""
        if context.question_category in ['STAR', 'Behavioral']:
            if content_analysis['star_method_usage']:
                return "Excellent use of STAR method! Your answer is well-structured and comprehensive."
            else:
                return "Consider using the STAR method: Situation (context), Task (your role), Action (what you did), Result (outcome)."
        return None
    
    def _generate_emotion_coaching(self, emotion_analysis: Dict[str, Any]) -> Optional[str]:
        """Generate emotion-specific coaching advice"""
        if emotion_analysis['confidence_level'] == 'low':
            return "Take deep breaths before answering. Remember: you're prepared and capable. Practice power poses to boost confidence."
        elif emotion_analysis['confidence_level'] == 'high':
            return "Great confidence! Your enthusiasm shows through. Keep this positive energy while maintaining professionalism."
        return None
    
    def _generate_next_steps(self, overall_score: float) -> List[str]:
        """Generate next steps based on performance"""
        if overall_score >= 8.0:
            return [
                "Practice with more challenging questions",
                "Focus on refining your delivery and pacing",
                "Record yourself to identify subtle improvements"
            ]
        elif overall_score >= 6.0:
            return [
                "Practice STAR method with behavioral questions",
                "Work on adding more specific examples",
                "Practice speaking clearly and at a good pace"
            ]
        else:
            return [
                "Start with basic behavioral questions",
                "Practice the STAR method structure",
                "Work on building confidence through preparation"
            ]
    
    def _generate_confidence_boost(self, overall_score: float) -> str:
        """Generate encouraging confidence boost message"""
        if overall_score >= 8.0:
            return "🎉 Outstanding performance! You're interview-ready and will impress any hiring manager!"
        elif overall_score >= 6.0:
            return "👍 Great work! You're on the right track. With a bit more practice, you'll be unstoppable!"
        else:
            return "💪 Every expert was once a beginner. You're building valuable skills that will serve you well!"
    
    def _generate_fallback_feedback(self, context: CoachingContext) -> CoachingFeedback:
        """Generate fallback feedback if analysis fails"""
        return CoachingFeedback(
            overall_score=5.0,
            content_score=5.0,
            emotion_score=5.0,
            delivery_score=5.0,
            strengths=["You completed the practice session"],
            areas_for_improvement=["Technical analysis unavailable"],
            specific_suggestions=["Try recording your answer again"],
            star_method_feedback="Use STAR method: Situation, Task, Action, Result",
            emotion_coaching="Practice speaking with confidence and clarity",
            next_steps=["Re-record your answer", "Check your microphone settings"],
            confidence_boost="Keep practicing - every attempt makes you better!"
        )
