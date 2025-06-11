# TEXT-SUMMARIZARIZATION-TOOL

"COMPANY" :CODTECH IT SOLUTIONS

"NAME": S HEMANTH

"INTERN ID" : CT06DN1480

"DOMAIN" : ARTIFICIAL INTELLIGENCE

"DURATION" :  6 WEEKS

"MENTOR" : NEELA SANTHOSH

## DESCRIPTION

This Python program is a command-line tool designed for automatic text summarization using the TextRank algorithm, which is a graph-based ranking model inspired by Google's PageRank. The goal of the tool is to extract the most meaningful and relevant sentences from a given input text, allowing users to quickly grasp the essential points of large documents. It leverages several important libraries such as nltk (Natural Language Toolkit) for text processing, numpy for numerical operations, networkx for building and analyzing the similarity graph, and sklearn for computing TF-IDF (Term Frequency-Inverse Document Frequency) representations used in sentence similarity calculations.

The program starts by importing the necessary libraries and downloading key resources from the NLTK library, specifically the punkt tokenizer and English stopwords. It’s important to note that the line nltk.download('punkt_tab') in the code is incorrect and should be removed because 'punkt_tab' is not a valid resource; only 'punkt' and 'stopwords' are required. The core logic begins with preprocessing the input text. The preprocess_text function tokenizes the input paragraph into individual sentences. Each sentence is then further tokenized into words, converted to lowercase, and stripped of common stopwords and non-alphabetic characters to produce a cleaned list of sentences that will be used for similarity analysis.

Next, the build_similarity_matrix function creates a cosine similarity matrix between these cleaned sentences. This is done by converting the sentences into TF-IDF vectors using TfidfVectorizer from sklearn. Cosine similarity is then computed between each pair of sentence vectors to form a numerical matrix that indicates how similar each sentence is to the others. This matrix is converted into a graph where each sentence is represented as a node and the similarity score between sentences becomes the edge weight connecting them.

The textrank_summarizer function then applies the PageRank algorithm to this graph using the pagerank function from the networkx library. PageRank assigns a score to each sentence based on its importance in the graph. Sentences with higher scores are considered more important. The top num_sentences (default is 3) with the highest scores are selected as the summary. These are extracted from the original, unprocessed sentences to preserve their original form and clarity.

In the final part of the program, the script runs as a standalone tool. It prompts the user to enter a paragraph, ending input with the word "END" on a new line. Once the input is collected, the program processes it using the TextRank summarization algorithm and prints out the top-ranked sentences as the final summary. This makes the tool user-friendly and suitable for educational or research purposes where quick summaries are needed. The method is unsupervised and doesn’t rely on any pre-labeled data, making it broadly applicable across different domains and text types. Overall, this summarizer offers a lightweight and effective approach to condensing text information.



# OUTPUT
![Image](https://github.com/user-attachments/assets/3c34d8bb-29ca-4b83-a7f1-8a0dd5f6dcdd)
