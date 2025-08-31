#!/usr/bin/env python3
"""
Simple RAG Retriever for Interview Questions
============================================

A lightweight retrieval system that loads interview questions from JSON files
and performs keyword-based search to find relevant questions for coaching.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SimpleRAGRetriever:
    """Simple retrieval system for interview questions"""
    
    def __init__(self, data_folder: str = "data/"):
        """
        Initialize the RAG retriever
        
        Args:
            data_folder: Path to folder containing JSON question files
        """
        self.data_folder = Path(data_folder)
        self.questions_database = {}
        self.question_metadata = []
        
        # Load all question files
        self._load_questions()
    
    def _load_questions(self) -> None:
        """Load all question files from the data folder"""
        try:
            if not self.data_folder.exists():
                logger.warning(f"Data folder {self.data_folder} does not exist")
                return
            
            # Find all JSON files
            json_files = list(self.data_folder.glob("*.json"))
            logger.info(f"Found {len(json_files)} JSON files in {self.data_folder}")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract questions from different file structures
                    questions = self._extract_questions(data, json_file.name)
                    
                    # Store questions with metadata
                    for question_data in questions:
                        self.question_metadata.append({
                            'file': json_file.name,
                            'category': question_data.get('category', 'Unknown'),
                            'question': question_data['question'],
                            'answer': question_data.get('answer', ''),
                            'keywords': self._extract_keywords(question_data['question'])
                        })
                    
                    logger.info(f"Loaded {len(questions)} questions from {json_file.name}")
                    
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {e}")
            
            logger.info(f"Total questions loaded: {len(self.question_metadata)}")
            
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
    
    def _extract_questions(self, data: Dict, filename: str) -> List[Dict]:
        """Extract questions from different JSON structures"""
        questions = []
        
        # Handle different file structures
        if "STAR_Questions" in data:
            for q in data["STAR_Questions"]:
                questions.append({
                    'category': 'STAR',
                    'question': q['question'],
                    'answer': q.get('answer', '')
                })
        
        if "Behavioral_Questions" in data:
            for q in data["Behavioral_Questions"]:
                questions.append({
                    'category': 'Behavioral',
                    'question': q['question'],
                    'answer': q.get('answer', '')
                })
        
        if "Technical_Questions" in data:
            for q in data["Technical_Questions"]:
                questions.append({
                    'category': 'Technical',
                    'question': q['question'],
                    'answer': q.get('answer', '')
                })
        
        # Handle nested structures (like AI_ML_Engineer_Interview_Preparation)
        for key, value in data.items():
            if isinstance(value, dict) and any(subkey in value for subkey in ["STAR_Questions", "Behavioral_Questions", "Technical_Questions"]):
                for subkey in ["STAR_Questions", "Behavioral_Questions", "Technical_Questions"]:
                    if subkey in value:
                        for q in value[subkey]:
                            questions.append({
                                'category': subkey.replace('_', ' ').title(),
                                'question': q['question'],
                                'answer': q.get('answer', '')
                            })
        
        return questions
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from question text"""
        # Simple keyword extraction - remove common words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        # Clean and split text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def search_questions(self, query: str, top_k: int = 5, category_filter: Optional[str] = None) -> List[Dict]:
        """
        Search for relevant questions based on query
        
        Args:
            query: Search query (user's answer or topic)
            top_k: Number of top results to return
            category_filter: Optional category filter (STAR, Behavioral, Technical)
            
        Returns:
            List of relevant questions with relevance scores
        """
        if not self.question_metadata:
            logger.warning("No questions loaded")
            return []
        
        # Extract keywords from query
        query_keywords = self._extract_keywords(query.lower())
        
        # Score each question based on keyword overlap
        scored_questions = []
        
        for question_data in self.question_metadata:
            # Skip if category filter is applied
            if category_filter and category_filter.lower() not in question_data['category'].lower():
                continue
            
            # Calculate relevance score
            score = self._calculate_relevance_score(query_keywords, question_data['keywords'])
            
            scored_questions.append({
                'question': question_data['question'],
                'answer': question_data['answer'],
                'category': question_data['category'],
                'file': question_data['file'],
                'relevance_score': score
            })
        
        # Sort by relevance score (descending) and return top_k
        scored_questions.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return scored_questions[:top_k]
    
    def _calculate_relevance_score(self, query_keywords: List[str], question_keywords: List[str]) -> float:
        """Calculate relevance score between query and question"""
        if not query_keywords or not question_keywords:
            return 0.0
        
        # Count keyword matches
        matches = sum(1 for qk in query_keywords if any(qk in qqk or qqk in qk for qqk in question_keywords))
        
        # Calculate Jaccard similarity
        union = len(set(query_keywords + question_keywords))
        if union == 0:
            return 0.0
        
        jaccard = matches / union
        
        # Boost score for exact matches
        exact_matches = sum(1 for qk in query_keywords if qk in question_keywords)
        exact_boost = exact_matches * 0.2
        
        return min(1.0, jaccard + exact_boost)
    
    def get_question_by_category(self, category: str, limit: int = 10) -> List[Dict]:
        """Get questions by specific category"""
        category_questions = [
            q for q in self.question_metadata 
            if category.lower() in q['category'].lower()
        ]
        return category_questions[:limit]
    
    def get_random_questions(self, count: int = 5) -> List[Dict]:
        """Get random questions for practice"""
        import random
        if len(self.question_metadata) <= count:
            return self.question_metadata
        
        return random.sample(self.question_metadata, count)
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded questions"""
        if not self.question_metadata:
            return {'total_questions': 0, 'categories': {}, 'files': {}}
        
        stats = {
            'total_questions': len(self.question_metadata),
            'categories': {},
            'files': {}
        }
        
        # Count by category
        for q in self.question_metadata:
            category = q['category']
            stats['categories'][category] = stats['categories'].get(category, 0) + 1
            
            file = q['file']
            stats['files'][file] = stats['files'].get(file, 0) + 1
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    # Test the retriever
    retriever = SimpleRAGRetriever()
    
    print("=== RAG Retriever Test ===")
    print(f"Total questions loaded: {len(retriever.question_metadata)}")
    
    # Get statistics
    stats = retriever.get_statistics()
    print(f"Categories: {stats['categories']}")
    print(f"Files: {stats['files']}")
    
    # Test search
    test_query = "machine learning model optimization"
    results = retriever.search_questions(test_query, top_k=3)
    
    print(f"\n=== Search Results for: '{test_query}' ===")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['category']}] {result['question']}")
        print(f"   Score: {result['relevance_score']:.3f}")
        print(f"   File: {result['file']}")
        print()
