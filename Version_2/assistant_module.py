from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
import json
import pickle
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

template = """
You are an AI Virtual Assistant.
- Your name is 'Neo'.
- You are just an Ai Assistant who helps user with there query.
- Try to be more casual and friendly.
- Generate friendly, humorous and natural human like responses.
- Keep your responses brief and precise.

Here is the conversation history: {context}

User input: {question}
"""

class VirtualAssistant:
    def __init__(self, confidence_threshold = 0.7):
        self.confidence_threshold = confidence_threshold
        self.Vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.classifier = SVC(kernel='linear', probability=True)
        self.pipeline = None
        self.intents = {}