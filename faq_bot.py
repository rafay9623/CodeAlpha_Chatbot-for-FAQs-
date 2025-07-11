from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Enhanced FAQ dataset
faqs = [
    {
    "question": "What is AI?",
    "answer": "AI stands for Artificial Intelligence. It is the simulation of human intelligence in machines that are programmed to think and act like humans."
},
{
    "question": "Define artificial intelligence",
    "answer": "Artificial Intelligence (AI) is a branch of computer science focused on building smart machines capable of performing tasks that typically require human intelligence."
},
{
    "question": "Can you explain AI?",
    "answer": "AI refers to machines or systems that mimic human intelligence to perform tasks such as learning, reasoning, and problem-solving."
},

    {"question": "What is Artificial Intelligence?", 
     "answer": "AI refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions."},

    {"question": "What are the main types of AI?", 
     "answer": "The main types are Narrow AI (specialized), General AI (human-level), and Superintelligent AI (beyond human capabilities, theoretical)."},

    {"question": "How does machine learning work?", 
     "answer": "It works by feeding data to algorithms that learn patterns and make decisions or predictions without being explicitly programmed."},

    {"question": "Difference between AI and machine learning?", 
     "answer": "AI is a broader concept; machine learning is a subset of AI focused on training models from data."},

    {"question": "What are real-world applications of AI?", 
     "answer": "Applications include chatbots, facial recognition, autonomous vehicles, fraud detection, healthcare diagnostics, and recommendation engines."},

    {"question": "What are neural networks?", 
     "answer": "Neural networks are algorithms inspired by the human brain structure that learn complex patterns from data."},

    {"question": "What is deep learning?", 
     "answer": "Deep learning is a branch of machine learning using multi-layered neural networks to learn from vast amounts of data."},

    {"question": "What is natural language processing?", 
     "answer": "NLP is a field of AI that enables machines to understand, interpret, and generate human language."},

    {"question": "What is computer vision?", 
     "answer": "Computer vision is a field of AI that trains machines to interpret and understand visual data like images and videos."},

    {"question": "What programming languages are used in AI?", 
     "answer": "Popular languages include Python, R, Java, C++, and Julia, with Python being the most widely used."},

    {"question": "What are the ethical concerns of AI?", 
     "answer": "They include privacy, bias in algorithms, job displacement, accountability, and potential misuse like surveillance or autonomous weapons."},

    {"question": "How is AI used in healthcare?", 
     "answer": "AI is used for diagnosis, predicting disease, personalized treatment, medical imaging, drug discovery, and robotic surgeries."},

    {"question": "Can AI replace humans?", 
     "answer": "AI can automate certain tasks but lacks general human intelligence and creativity. It complements rather than replaces humans in many fields."},

    {"question": "What skills are required to learn AI?", 
     "answer": "You need math (especially linear algebra and calculus), programming (Python), data handling, machine learning, and problem-solving skills."}
]

# Preprocess questions using spaCy
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Prepare vectorized FAQ dataset
faq_questions = [faq["question"] for faq in faqs]
processed_questions = [preprocess(q) for q in faq_questions]
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)

# Function to get best match
def get_best_match(user_input):
    processed_input = preprocess(user_input)
    input_vector = vectorizer.transform([processed_input])
    similarities = cosine_similarity(input_vector, faq_vectors)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    if best_score > 0.2:
        return faqs[best_idx]["answer"], round(float(best_score), 2)
    else:
        return "Sorry, I couldn't find a relevant answer. Please try rephrasing.", 0

# Flask route
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    answer, score = get_best_match(question)
    return jsonify({"answer": answer, "score": score})

if __name__ == '__main__':
    app.run(debug=True)
