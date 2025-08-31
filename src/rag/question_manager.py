#!/usr/bin/env python3
"""
Question Manager for Interview Practice
======================================

Manages question cycling, prevents repeats, and ensures random ordering
for interview practice sessions.
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime

logger = logging.getLogger(__name__)

class QuestionManager:
    """Manages interview questions with cycling and no-repeat logic"""
    
    def __init__(self, data_folder: str = "data/"):
        """
        Initialize the question manager
        
        Args:
            data_folder: Path to folder containing JSON question files
        """
        self.data_folder = Path(data_folder)
        self.questions_database = {}
        self.question_metadata = []
        self.asked_questions: Set[str] = set()
        self.current_cycle = 1
        self.current_file_type = None
        self.question_order = []
        
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
                    
                    # Store questions with metadata and unique IDs
                    for question_data in questions:
                        question_id = self._generate_question_id(json_file.name, question_data)
                        self.question_metadata.append({
                            'id': question_id,
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
    
    def _generate_question_id(self, filename: str, question_data: Dict) -> str:
        """Generate unique question ID"""
        # Extract file type (e.g., "data_analyst" from "data_analyst.json")
        file_type = filename.replace('.json', '').lower()
        
        # Extract category (e.g., "star" from "STAR")
        category = question_data['category'].lower().replace(' ', '_')
        
        # Generate a hash-based ID to ensure uniqueness
        import hashlib
        question_hash = hashlib.md5(question_data['question'].encode()).hexdigest()[:8]
        
        return f"{file_type}_{category}_{question_hash}"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from question text"""
        # Simple keyword extraction - remove common words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        # Clean and split text
        import re
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def select_file_type(self, file_type: str) -> bool:
        """
        Select a specific file type for question cycling
        
        Args:
            file_type: Type of questions (e.g., "data_analyst", "ai_ml_engineer")
            
        Returns:
            True if file type exists and questions are available
        """
        # Check if file exists
        file_path = self.data_folder / f"{file_type}.json"
        if not file_path.exists():
            logger.error(f"File type {file_type} not found: {file_path}")
            return False
        
        # Filter questions for this file type
        file_questions = [q for q in self.question_metadata if q['file'] == f"{file_type}.json"]
        
        if not file_questions:
            logger.error(f"No questions found for file type {file_type}")
            return False
        
        # Set current file type and reset cycle
        self.current_file_type = file_type
        self.asked_questions.clear()
        self.current_cycle = 1
        
        # Create shuffled question order for this cycle
        self.question_order = [q['id'] for q in file_questions]
        random.shuffle(self.question_order)
        
        logger.info(f"Selected file type: {file_type} with {len(file_questions)} questions")
        return True
    
    def get_next_question(self) -> Optional[Dict]:
        """
        Get the next unasked question
        
        Returns:
            Question data or None if all questions have been asked
        """
        if not self.current_file_type:
            logger.warning("No file type selected. Call select_file_type() first.")
            return None
        
        # Get questions for current file type
        file_questions = [q for q in self.question_metadata if q['file'] == f"{self.current_file_type}.json"]
        
        # Find unasked questions
        unasked_questions = [q for q in file_questions if q['id'] not in self.asked_questions]
        
        if not unasked_questions:
            # All questions asked, start new cycle
            self._start_new_cycle()
            unasked_questions = file_questions
        
        # Get random unasked question
        selected_question = random.choice(unasked_questions)
        
        # Mark as asked
        self.asked_questions.add(selected_question['id'])
        
        logger.info(f"Selected question: {selected_question['id']} (Cycle {self.current_cycle})")
        return selected_question
    
    def _start_new_cycle(self) -> None:
        """Start a new cycle with shuffled question order"""
        self.current_cycle += 1
        self.asked_questions.clear()
        
        # Get questions for current file type
        file_questions = [q for q in self.question_metadata if q['file'] == f"{self.current_file_type}.json"]
        
        # Create new shuffled order
        self.question_order = [q['id'] for q in file_questions]
        random.shuffle(self.question_order)
        
        logger.info(f"Started new cycle {self.current_cycle} with {len(file_questions)} questions")
    
    def get_progress(self) -> Dict:
        """Get current progress information"""
        if not self.current_file_type:
            return {
                'file_type': None,
                'current_cycle': 0,
                'total_questions': 0,
                'asked_questions': 0,
                'remaining_questions': 0,
                'progress_percentage': 0.0
            }
        
        # Get questions for current file type
        file_questions = [q for q in self.question_metadata if q['file'] == f"{self.current_file_type}.json"]
        total_questions = len(file_questions)
        asked_count = len(self.asked_questions)
        remaining = total_questions - asked_count
        
        progress_percentage = (asked_count / total_questions * 100) if total_questions > 0 else 0.0
        
        return {
            'file_type': self.current_file_type,
            'current_cycle': self.current_cycle,
            'total_questions': total_questions,
            'asked_questions': asked_count,
            'remaining_questions': remaining,
            'progress_percentage': progress_percentage
        }
    
    def get_available_file_types(self) -> List[str]:
        """Get list of available file types"""
        json_files = list(self.data_folder.glob("*.json"))
        return [f.stem for f in json_files]  # Remove .json extension
    
    def reset_session(self) -> None:
        """Reset the current session (clear asked questions, reset cycle)"""
        self.asked_questions.clear()
        self.current_cycle = 1
        self.current_file_type = None
        self.question_order = []
        logger.info("Session reset")
    
    def get_question_by_id(self, question_id: str) -> Optional[Dict]:
        """Get question by its unique ID"""
        for question in self.question_metadata:
            if question['id'] == question_id:
                return question
        return None
    
    def mark_question_asked(self, question_id: str) -> bool:
        """Manually mark a question as asked"""
        if question_id in self.asked_questions:
            return False  # Already asked
        
        self.asked_questions.add(question_id)
        logger.info(f"Manually marked question as asked: {question_id}")
        return True

# Example usage and testing
if __name__ == "__main__":
    # Test the question manager
    manager = QuestionManager()
    
    print("=== Question Manager Test ===")
    print(f"Available file types: {manager.get_available_file_types()}")
    
    # Select a file type
    if manager.select_file_type("data_analyst"):
        print("✅ Selected data_analyst file type")
        
        # Get progress
        progress = manager.get_progress()
        print(f"Progress: {progress}")
        
        # Get a few questions
        for i in range(3):
            question = manager.get_next_question()
            if question:
                print(f"\nQuestion {i+1}: {question['question'][:100]}...")
                print(f"ID: {question['id']}")
                print(f"Category: {question['category']}")
        
        # Check progress again
        progress = manager.get_progress()
        print(f"\nUpdated Progress: {progress}")
    else:
        print("❌ Failed to select data_analyst file type")
