"""
RAG (Retrieval-Augmented Generation) Package

This package provides simple retrieval capabilities for interview questions
and coaching content.
"""

from .simple_retriever import SimpleRAGRetriever
from .question_manager import QuestionManager

__all__ = ['SimpleRAGRetriever', 'QuestionManager']
__version__ = '0.1.0'
