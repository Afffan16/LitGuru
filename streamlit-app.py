import pandas as pd
import numpy as np
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==============================
# Load env vars
# ==============================
load_dotenv()
openai_api = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", ""))

# ==============================
# Cache decorators
# ==============================
@st.cache_data
def load_book_data():
    books = pd.read_csv("books_with_emotions.csv")
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "cover-not-found.jpg",
        books["large_thumbnail"],
    )
    return books

@st.cache_resource
def setup_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

@st.cache_resource
def setup_chroma(_embeddings):
    # Initialize ChromaDB with minimal settings
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=_embeddings
    )
    
    try:
        # Check if collection needs initialization
        collection_size = len(db.get()["ids"])
        if collection_size == 0:
            books_data = load_book_data()
            texts = []
            for _, row in books_data.iterrows():
                text = f"{row['isbn13']} {row['title']} {row['authors']} {row['description']}"
                texts.append(text)
            
            # Add documents to ChromaDB
            db.add_texts(texts=texts)
            db.persist()
    except Exception as e:
        st.warning(f"Collection initialization skipped: {str(e)}")
    
    return db

# ==============================
# Initialize components
# ==============================
books = load_book_data()
huggingface_embeddings = setup_embeddings()
db_books = setup_chroma(huggingface_embeddings)

# ==============================
# Recommendation Logic
# ==============================

def retrieve_semantic_recommendations(query: str, initial_top_k: int = 50) -> pd.DataFrame:
    try:
        # Get recommendations with metadata
        results = db_books.similarity_search_with_score(
            query, 
            k=initial_top_k
        )
        # Extract ISBNs from results
        books_list = [int(doc.page_content.split()[0]) for doc, _ in results]
        return books[books["isbn13"].isin(books_list)]
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return pd.DataFrame()

def recommend_books(query: str, category: str, tone: str, final_top_k: int = 16):
    # Get recommendations
    title_matches = books[books["title"].str.contains(query, case=False, na=False)]
    semantic_matches = retrieve_semantic_recommendations(query)
    
    # Combine results
    recommendations = pd.concat([title_matches, semantic_matches]).drop_duplicates("isbn13")
    
    # Apply filters
    if category != "All":
        recommendations = recommendations[recommendations["simple_categories"] == category]
    
    # Sort by emotional tone
    if tone == "Happy":
        recommendations = recommendations.nlargest(final_top_k, 'joy')
    elif tone == "Surprising":
        recommendations = recommendations.nlargest(final_top_k, 'surprise')
    elif tone == "Angry":
        recommendations = recommendations.nlargest(final_top_k, 'anger')
    elif tone == "Suspenseful":
        recommendations = recommendations.nlargest(final_top_k, 'fear')
    elif tone == "Sad":
        recommendations = recommendations.nlargest(final_top_k, 'sadness')
    
    return recommendations.head(final_top_k)



# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="📚 LitGuru", layout="wide")
st.title("📚 LitGuru")
st.markdown(
    "⚡ **Sparking Your Next Great Read with Smart, Lightning-Fast Book Matches!**"
)

# Sidebar filters
st.sidebar.header("Filters")
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

category = st.sidebar.selectbox("Category", categories)
tone = st.sidebar.selectbox("Tone", tones)

# ==============================
# Search bar with clear button
# ==============================

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

if "search_input_key" not in st.session_state:
    st.session_state.search_input_key = "search_input_1"

search_col, clear_col = st.columns([20, 1])
with search_col:
    st.session_state.search_query = st.text_input(
        "🔍 Search for a book (title, author, description):",
        value=st.session_state.search_query,
        key=st.session_state.search_input_key
    )

with clear_col:
    st.write("") 
    st.write("")  
    if st.button("✖️"):
        st.session_state.search_query = ""
        st.session_state.recs = []
        st.session_state.selected_books = {}
        
        if st.session_state.search_input_key == "search_input_1":
            st.session_state.search_input_key = "search_input_2"
        else:
            st.session_state.search_input_key = "search_input_1"
        st.rerun()

# ==============================
# Session state
# ==============================
if "selected_books" not in st.session_state:
    st.session_state.selected_books = {}  # dict of isbn13 -> book dict
if "recs" not in st.session_state:
    st.session_state.recs = []

# ==============================
# Search & Load Recommendations
# ==============================

search_triggered = st.button("Find Recommendations", key="search_button")

if st.session_state.search_query and (search_triggered or st.session_state.search_query != st.session_state.get("last_query", "")):
    st.session_state.last_query = st.session_state.search_query
    recs = recommend_books(st.session_state.search_query, category, tone)
    st.session_state.recs = recs.to_dict(orient="records")


# ==============================
# Layout: Main + Right Column for Details
# ==============================
main_col, right_col = st.columns([3, 1])

# --- Recommendations Grid ---
with main_col:
    if st.session_state.recs:
        st.subheader("📚 Recommended Books")
        cols = st.columns(4)
        for i, row in enumerate(st.session_state.recs):
            with cols[i % 4]:
                st.image(row["large_thumbnail"], use_container_width=True)
                st.markdown(f"**{row['title']}**")
                st.caption(f"👤 {row['authors']}")
                desc = " ".join(str(row["description"]).split()[:25]) + "..."
                st.caption(desc)
                if st.button("📖 View Details", key=f"btn_{row['isbn13']}"):
                    st.session_state.selected_books[row['isbn13']] = row

# --- Right Column: Book Details ---
with right_col:
    if st.session_state.selected_books:
        st.subheader("📖 Book Details")
        # Loop through all selected books
        for isbn, book in list(st.session_state.selected_books.items()):
            st.markdown("---")
            st.image(book["large_thumbnail"], width=200)
            st.markdown(f"**{book['title']}**")
            st.markdown(f"👤 **Author(s):** {book['authors']}")
            st.markdown(f"🏷️ **Category:** {book['simple_categories']}")
            st.markdown("### 📄 Description")
            st.write(book["description"])
            if st.button(f"🔙 Close", key=f"close_{isbn}"):
                del st.session_state.selected_books[isbn]
