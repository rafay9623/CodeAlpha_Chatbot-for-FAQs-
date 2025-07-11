# CodeAlpha_Chatbot-for-FAQs-

AI FAQ Bot

A simple FAQ bot that answers AI-related questions using NLP with a Flask backend and HTML frontend.

Features
Answers AI questions from a predefined FAQ dataset.
Uses spaCy and TF-IDF for question matching.
Basic web interface for user queries.

Prerequisites
Python 3.8+
Install: flask, flask-cors, spacy, scikit-learn, numpy
spaCy model: python -m spacy download en_core_web_sm

Installation
Clone repo: git clone <repository-url>
Install dependencies: pip install -r requirements.txt
Download spaCy model: python -m spacy download en_core_web_sm

Usage
Run backend: python faq_bot.py
Open faq_bot_frontend.html in a browser or serve it: python -m http.server 8000
Ask AI-related questions via the web interface.

Files
faq_bot.py: Flask backend with NLP logic.
faq_bot_frontend.html: HTML frontend.
README.md: This file.
