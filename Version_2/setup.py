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

    def load_training_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        x = []
        y = []

        for intent, content in data.items():
            self.intents[intent] = content.get('actions', [])
            for phrase in content.get('training_phrases', []):
                x.append(phrase)
                y.append(intent)

        return x,y
    
    def train(self, training_data_path):
        x,y = self.load_training_data(training_data_path)
        
        self.pipeline = Pipeline([
            ('vectorizer', self.Vectorizer),
            ('classifier', self.classifier)
        ])

        self.pipeline.fit(x,y)

    def classify_intent(self, user_input):
        if not self.pipeline:
            raise ValueError('Model not trained. Please train the model first.')

        probs = self.pipeline.predict_proba([user_input])[0]
        max_prob = np.max(probs)

        if max_prob >= self.confidence_threshold:
            intent = self.pipeline.classes_[np.argmax(probs)]

            return intent, max_prob

        return None, max_prob