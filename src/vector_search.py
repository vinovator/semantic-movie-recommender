# src/vector_search.py

import os
import streamlit as st
import chromadb
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection
from . import config
from scipy.spatial.distance import cosine 

class SearchEngine:
    def __init__(self):
        """
        Initializes the Search Engine with environment-aware ChromaDB client.
        - On Streamlit Cloud: Builds an in-memory database.
        - Locally: Uses a persistent on-disk database for speed.
        """
        print("Initializing Search Engine...")
        
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.df = pd.read_parquet(config.PROCESSED_FILE)
        
        # --- ENVIRONMENT-AWARE INITIALIZATION ---
        # Check for an environment variable set by Streamlit Cloud
        if "STREAMLIT_SERVER_RUNNING_ON_CLOUD" in os.environ:
            print("Running in Streamlit Cloud environment. Initializing in-memory ChromaDB.")
            # Use the Streamlit connection to get an in-memory client
            self.client = st.connection('chromadb', type=ChromadbConnection).client
            self._build_collections_in_memory()
        else:
            print("Running in local environment. Initializing persistent ChromaDB.")
            # Use the faster persistent client for local development
            self.client = chromadb.PersistentClient(path=config.DB_DIR)
            self.plot_collection = self.client.get_collection(name="movie_plots")
            self.setup_collection = self.client.get_collection(name="plot_setups")
            self.payoff_collection = self.client.get_collection(name="plot_payoffs")
        
        print("Search Engine Initialized Successfully.")

    def _build_collections_in_memory(self):
        """A helper method to create and populate in-memory collections for the cloud environment."""
        print("Building in-memory vector database... This may take a minute on first run.")
        self.plot_collection = self.client.get_or_create_collection(name="movie_plots")
        self.setup_collection = self.client.get_or_create_collection(name="plot_setups")
        self.payoff_collection = self.client.get_or_create_collection(name="plot_payoffs")

        # Check if collections are already populated to avoid rebuilding on every rerun
        if self.plot_collection.count() > 0:
            print("Collections already exist in memory.")
            return

        plots = self.df['plot'].tolist()
        ids = self.df['tconst'].tolist()
        setups = self.df['plot'].apply(lambda p: '. '.join(p.split('. ')[:int(len(p.split('. '))*0.3)]) + '.').tolist()
        payoffs = self.df['plot'].apply(lambda p: '. '.join(p.split('. ')[int(len(p.split('. '))*0.3):]) + '.').tolist()

        plot_embeddings = self.model.encode(plots, show_progress_bar=True)
        setup_embeddings = self.model.encode(setups, show_progress_bar=True)
        payoff_embeddings = self.model.encode(payoffs, show_progress_bar=True)
        
        metadata_df = self.df.drop(columns=['plot']).copy()
        numeric_cols = metadata_df.select_dtypes(include=np.number).columns
        metadata_df[numeric_cols] = metadata_df[numeric_cols].fillna(0)
        metadata_df = metadata_df.fillna("")
        metadata = metadata_df.to_dict('records')

        batch_size = 4096
        for i in range(0, len(ids), batch_size):
            end_i = min(i + batch_size, len(ids))
            self.plot_collection.add(ids=ids[i:end_i], embeddings=plot_embeddings[i:end_i].tolist(), metadatas=metadata[i:end_i])
            self.setup_collection.add(ids=ids[i:end_i], embeddings=setup_embeddings[i:end_i].tolist(), metadatas=metadata[i:end_i])
            self.payoff_collection.add(ids=ids[i:end_i], embeddings=payoff_embeddings[i:end_i].tolist(), metadatas=metadata[i:end_i])
        print("In-memory database built successfully.")

    # --- The search() and find_plot_twist() methods remain exactly the same ---
    def search(self, query_text=None, filters={}, k=config.TOP_K_RESULTS):
        # ... (no changes needed here)
        if query_text:
            query_embedding = self.model.encode(query_text).tolist()
            results = self.plot_collection.query(query_embeddings=[query_embedding], n_results=100)
            ids = results['ids'][0]
            if not ids:
                return pd.DataFrame()
            search_df = self.df[self.df['tconst'].isin(ids)].copy()
        else:
            search_df = self.df.copy()

        if filters:
            for key, value in filters.items():
                if not value:
                    continue
                if key == 'startYear':
                    search_df = search_df[search_df['startYear'].between(value[0], value[1])]
                elif key == 'averageRating':
                    search_df = search_df[search_df['averageRating'] >= value]
                else:
                    search_df = search_df[search_df[key].str.contains(value, case=False, na=False)]

        return search_df.sort_values(by='averageRating', ascending=False).head(k)
    
    def find_plot_twist(self, tconst, k=5):
        # ... (no changes needed here)
        setup_vectors = self.setup_collection.get(ids=[tconst], include=['embeddings'])
        if not setup_vectors['ids']:
            print(f"Warning: Could not find setup vector for tconst {tconst}")
            return pd.DataFrame()
        reference_setup_vector = setup_vectors['embeddings'][0]

        payoff_vectors = self.payoff_collection.get(ids=[tconst], include=['embeddings'])
        if not payoff_vectors['ids']:
            print(f"Warning: Could not find payoff vector for tconst {tconst}")
            return pd.DataFrame()
        reference_payoff_vector = payoff_vectors['embeddings'][0]

        similar_setup_results = self.setup_collection.query(query_embeddings=[reference_setup_vector], n_results=50)
        candidate_ids = similar_setup_results['ids'][0]
        if tconst in candidate_ids:
            candidate_ids.remove(tconst)
        if not candidate_ids:
            return pd.DataFrame()

        candidate_payoff_vectors = self.payoff_collection.get(ids=candidate_ids, include=['embeddings'])['embeddings']
        dissimilarities = []
        for candidate_vector in candidate_payoff_vectors:
            dist = cosine(reference_payoff_vector, candidate_vector)
            dissimilarities.append(dist)

        results_df = pd.DataFrame({'tconst': candidate_ids, 'dissimilarity': dissimilarities})
        top_k_ids = results_df.sort_values(by='dissimilarity', ascending=False).head(k)
        return self.df[self.df['tconst'].isin(top_k_ids['tconst'])]

# This block allows for direct testing of the module.
# To run, execute `python -m src.vector_search` from the root directory.
if __name__ == '__main__':
    engine = SearchEngine()
    
    print("\n--- Example 1: Pure Semantic Search ---")
    results = engine.search(query_text="a submarine crew must prevent a nuclear war")
    print(results[['primaryTitle', 'startYear', 'averageRating']])
    
    print("\n--- Example 2: Hybrid Search ---")
    hybrid_filters = {'directors': 'Christopher Nolan', 'startYear': (2010, 2020)}
    results = engine.search(query_text="a thief enters people's dreams", filters=hybrid_filters)
    print(results[['primaryTitle', 'startYear', 'directors', 'averageRating']])
    
    print("\n--- Example 3: Pure Metadata Search ---")
    metadata_filters = {'actors': 'Tom Hanks', 'genres': 'Drama', 'averageRating': 8.5}
    results = engine.search(filters=metadata_filters)
    print(results[['primaryTitle', 'startYear', 'actors', 'genres', 'averageRating']])