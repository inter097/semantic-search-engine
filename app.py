import os
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Page configuration
st.set_page_config(page_title="Semantic Search Engine", page_icon="🔍", layout="wide")

# Load SpaCy for lemmatization
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except:
        # If not found, try to load from local if it exists in another project, 
        # but better to just download it for a clean repo
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_nlp()

def preprocess_text(text):
    doc = nlp(text.lower())
    # Lemmatization and removing stopwords/punctuation
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return " ".join(tokens)

class SemanticSearch:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.filenames = []
        self.documents = []

    def fit(self, docs, names):
        self.documents = docs
        self.filenames = names
        processed_docs = [preprocess_text(doc) for doc in docs]
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_docs)

    def search(self, query, top_k=5):
        if self.tfidf_matrix is None:
            return []
        
        processed_query = preprocess_text(query)
        query_vec = self.vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top k indices
        results = []
        for idx in similarities.argsort()[::-1]:
            if similarities[idx] > 0:
                results.append({
                    "filename": self.filenames[idx],
                    "score": similarities[idx],
                    "content": self.documents[idx]
                })
            if len(results) >= top_k:
                break
        return results

# UI
def main():
    st.title("🔍 Semantic Document Search Engine")
    st.markdown("### Search through your documents using Vector Space Modeling and TF-IDF.")

    if 'engine' not in st.session_state:
        st.session_state.engine = SemanticSearch()
        st.session_state.indexed = False

    with st.sidebar:
        st.header("📂 Document Indexing")
        uploaded_files = st.file_uploader("Upload TXT files", type="txt", accept_multiple_files=True)
        
        if st.button("⚡ Index Documents"):
            if not uploaded_files:
                st.warning("Please upload some files first.")
            else:
                with st.spinner("Preprocessing and indexing..."):
                    docs = []
                    names = []
                    for uploaded_file in uploaded_files:
                        content = str(uploaded_file.read(), "utf-8")
                        docs.append(content)
                        names.append(uploaded_file.name)
                    
                    st.session_state.engine.fit(docs, names)
                    st.session_state.indexed = True
                    st.success(f"Indexed {len(names)} documents!")

    # Search section
    st.divider()
    query = st.text_input("🎯 Enter your search query:", placeholder="e.g., machine learning concepts")

    if st.button("🔍 Search") or query:
        if not st.session_state.indexed:
            st.error("Please index some documents in the sidebar first.")
        elif not query.strip():
            st.warning("Please enter a query.")
        else:
            results = st.session_state.engine.search(query)
            
            if not results:
                st.info("No matching documents found.")
            else:
                st.subheader(f"Top {len(results)} Results")
                for i, res in enumerate(results):
                    with st.expander(f"📄 {res['filename']} (Score: {res['score']:.4f})"):
                        st.progress(float(res['score']))
                        st.markdown("**Snippet:**")
                        # Show a snippet of the content
                        snippet = res['content'][:500] + "..." if len(res['content']) > 500 else res['content']
                        st.write(snippet)
                        if st.button(f"View Full Content", key=f"btn_{i}"):
                            st.text_area("Full Content", res['content'], height=300)

if __name__ == "__main__":
    main()
