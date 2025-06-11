# Required Libraries
import nltk
nltk.download('punkt_tab')
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data: tokenizer and stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Function to clean and preprocess text
def preprocess_text(text):
    # Split text into sentences
    sentences = sent_tokenize(text)

    # Load English stopwords
    stop_words = set(stopwords.words("english"))

    cleaned_sentences = []
    for sentence in sentences:
        # Tokenize and lowercase words
        words = word_tokenize(sentence.lower())
        # Remove stopwords and non-alphabetic tokens
        words = [word for word in words if word.isalpha() and word not in stop_words]
        # Join words back into cleaned sentence
        cleaned_sentences.append(" ".join(words))

    return sentences, cleaned_sentences

# Function to build similarity matrix between cleaned sentences
def build_similarity_matrix(sentences):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(sentences)
    # Cosine similarity = dot product of TF-IDF matrix with its transpose
    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    return similarity_matrix

# Main TextRank summarizer function
def textrank_summarizer(text, num_sentences=3):
    # Get original and cleaned sentences
    original_sentences, cleaned_sentences = preprocess_text(text)

    # If the text is too short, return original
    if len(original_sentences) < num_sentences:
        return text

    # Create similarity graph and apply PageRank
    similarity_matrix = build_similarity_matrix(cleaned_sentences)
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    # Rank sentences by their score
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(original_sentences)), reverse=True)
    summary_sentences = [s for _, s in ranked_sentences[:num_sentences]]

    # Return the selected top sentences as the final summary
    return " ".join(summary_sentences)

# --- Program Entry Point ---
if __name__ == "__main__":
    print(" TEXT SUMMARIZATION TOOL (TextRank Based)\n")
    print(" Paste your paragraph below. Type 'END' on a new line to finish input:\n")

    # Accept multiline input
    input_lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        input_lines.append(line)

    # Join all lines into a single string
    input_text = "\n".join(input_lines)

    # Generate summary
    print("\n Processing your text for summarization...\n")
    summary = textrank_summarizer(input_text, num_sentences=3)

    # Display structured output
    print(" --- SUMMARY RESULT --- \n")
    print(summary)
