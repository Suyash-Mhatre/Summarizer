import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    sentences = sent_tokenize(text)
    clean_sentences = [re.sub(r'\s+', ' ', re.sub('[^a-zA-Z]', ' ', sentence)) for sentence in sentences]
    clean_sentences = [sentence.lower() for sentence in clean_sentences]
    return sentences, clean_sentences

def sentence_embeddings(clean_sentences):
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    sentence_vectors = vectorizer.fit_transform(clean_sentences).toarray()
    return sentence_vectors

def rank_sentences(sentence_vectors):
    sim_matrix = cosine_similarity(sentence_vectors)
    sentence_scores = sim_matrix.sum(axis=1)
    ranked_sentences = [sentences[i] for i in np.argsort(sentence_scores, axis=0)[::-1]]
    return ranked_sentences

def summarize_text(ranked_sentences, num_sentences=3):
    summary = " ".join(ranked_sentences[:num_sentences])
    return summary

# Example usage
# text = """Your long text goes here..."""
text = input("""Your long text goes here : """)
sentences, clean_sentences = preprocess_text(text)
sentence_vectors = sentence_embeddings(clean_sentences)
ranked_sentences = rank_sentences(sentence_vectors)
summary = summarize_text(ranked_sentences, num_sentences=3)
print("summary :" + summary)
