# movie-review-intelligence-system
Movie Review Intelligence System — Sentiment Analysis + Semantic Similarity
📌 Overview

This project builds an NLP-based intelligence system that analyzes movie reviews along two complementary tracks:

Classical NLP — TF-IDF + Logistic Regression to classify review sentiment (Positive/Negative)
Modern NLP (Embeddings) — Sentence Transformers to generate dense vector representations and find semantically similar reviews using cosine similarity

The two approaches are combined into a single function that, given any new review, returns its predicted sentiment along with the most similar existing reviews — demonstrating both a traditional ML pipeline and a modern embedding-based semantic search pipeline side by side.

REVIEW → TEXT CLEANING
    ├── Classical Representation
    │       └── TF-IDF → Logistic Regression → Sentiment
    └── Embeddings
            └── Sentence Transformer → Vector → Cosine Similarity → Similar Reviews
🛠️ Project Workflow
1. Dataset

A small labeled sample of movie reviews (positive and negative) used to demonstrate both pipelines end-to-end.

2. Sentiment Classification (Classical NLP)
Converted reviews into numerical features using TF-IDF (with unigrams + bigrams)
Trained a Logistic Regression classifier on the TF-IDF vectors
Evaluated the model on unseen sample reviews to predict Positive/Negative sentiment
3. Semantic Similarity (Embeddings)
Generated dense sentence embeddings using the pretrained sentence-transformers/all-MiniLM-L6-v2 model
Encoded a new user review into the same embedding space
Computed cosine similarity between the user review and all existing reviews
Retrieved the top-k most semantically similar reviews — enabling semantic search without a vector database
4. Combined Analysis Function

Built a unified analyze_review() function that, for any input review:

Predicts sentiment using the TF-IDF + Logistic Regression pipeline
Computes its embedding and finds the most similar reviews via cosine similarity
Returns a combined result: sentiment label + top similar reviews with similarity scores
USER REVIEW
    ↓
Text Processing
    ↓
 ┌───────────────┐                 ┌────────────────────────┐
 │     TF-IDF    │                 │   Sentence Embedding   │
 │ Logistic Reg. │                 │     Dense Vector       │
 │   Sentiment   │                 │   Cosine Similarity    │
 └───────────────┘                 │  Top Similar Reviews   │
                                    └────────────────────────┘
    ↓
FINAL ANALYSIS
🧰 Tech Stack
Python
Scikit-learn – TF-IDF vectorization, Logistic Regression, cosine similarity
Sentence-Transformers – pretrained embedding model (all-MiniLM-L6-v2)
NumPy – similarity ranking and array operations
🚀 How to Run
Clone the repository
Install dependencies: pip install scikit-learn sentence-transformers numpy
Run the notebook nlp_final_project.ipynb
Call analyze_review("your review text here") to get sentiment + similar reviews
📈 Key Insights
Combining a classical bag-of-words model with modern embedding-based similarity gives both an interpretable sentiment label and rich semantic context
TF-IDF + Logistic Regression performs well for sentiment classification even on small datasets
Sentence embeddings enable semantic search that goes beyond keyword matching — capturing meaning and paraphrasing
This architecture is a lightweight foundation for building recommendation or review-intelligence systems without needing a dedicated vector database
