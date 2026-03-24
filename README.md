# 🔍 Semantic Search Engine

A document search tool that uses **Vector Space Modeling (VSM)** and **TF-IDF** (Term Frequency-Inverse Document Frequency) to find the most relevant contents from a document collection.

## 🚀 Features
- **Semantic Analysis**: Goes beyond keyword matching by using TF-IDF to weight important terms.
- **Natural Language Preprocessing**: Integrated **SpaCy** for lemmatization and stop-word removal, ensuring that search is insensitive to case, tense, or plural forms.
- **Top-K Ranking**: Displays the most relevant results ranked by **Cosine Similarity**.
- **Interactive Web Interface**: Built with **Streamlit** for a seamless user experience.
- **Snippet Highlighting**: Preview document contents with relevance scores and interactive expansion for full viewing.

## 🛠️ Technology Stack
- **Python**: Core engine.
- **Streamlit**: Web-based frontend.
- **Scikit-learn**: TF-IDF Vectorization and Cosine Similarity.
- **SpaCy**: Linguistic preprocessing (Lemmatization).
- **Pandas/NumPy**: Data manipulation.

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd SemanticSearchEngine
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 🧠 How it Works
The search engine treats documents and queries as vectors in a high-dimensional space (VSM). 
1. **Lemmatization**: Tokens are reduced to their dictionary form (e.g., *running* -> *run*).
2. **TF-IDF Weighting**: The system reduces the influence of common words and highlights the specialty of terms within each document.
3. **Cosine Similarity**: The mathematical "angle" between the query vector and document vectors determines the final relevance score.

---
*Developed as part of the Text Mining course portfolio.*
