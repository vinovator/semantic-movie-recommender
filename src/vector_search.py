# src/vector_search.py

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from . import config
from scipy.spatial.distance import cosine 
import numpy as np 

class SearchEngine:
    def __init__(self):
        """
        Initializes the Search Engine by loading the embedding model,
        the main data file, and connecting to the vector database.
        """
        print("Initializing Search Engine...")
        
        # Load the sentence transformer model specified in config
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Load the processed movie data from the Parquet file
        self.df = pd.read_parquet(config.PROCESSED_FILE)
        
        # Initialize a persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=config.DB_DIR)
        
        # Get the collection for movie plots. This is the primary collection for search.
        self.plot_collection = self.client.get_collection(name="movie_plots")

        # The following collections are for advanced features like the "Plot Twist Finder".
        # They are loaded here but used by separate methods that can be added later.
        self.setup_collection = self.client.get_collection(name="plot_setups")
        self.payoff_collection = self.client.get_collection(name="plot_payoffs")
        
        print("Search Engine Initialized Successfully.")


    def search(self, query_text=None, filters={}, k=config.TOP_K_RESULTS):
        """
        Performs a hybrid search using semantic text search and metadata filtering.
        """
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
                
                # --- SIMPLIFIED LOGIC ---
                # Now that actors/directors are strings, we can use one method for all text filters.
                if key == 'startYear':
                    search_df = search_df[search_df['startYear'].between(value[0], value[1])]
                elif key == 'averageRating':
                    search_df = search_df[search_df['averageRating'] >= value]
                else: # For string columns like 'genres', 'actors', and 'directors'
                    search_df = search_df[search_df[key].str.contains(value, case=False, na=False)]
                # --- END OF SIMPLIFICATION ---

        return search_df.sort_values(by='averageRating', ascending=False).head(k)
    
    
    def find_plot_twist(self, tconst, k=5):
        """
        Finds movies that start similarly but have different endings.
        """
        # 1. Get and validate the reference setup vector
        setup_vectors = self.setup_collection.get(ids=[tconst], include=['embeddings'])
        
        # --- FINAL FIX: Simplify the check to only use the 'ids' list ---
        # This avoids the ValueError by not evaluating the embeddings array directly.
        if not setup_vectors['ids']:
            print(f"Warning: Could not find setup vector for tconst {tconst}")
            return pd.DataFrame()
        reference_setup_vector = setup_vectors['embeddings'][0]

        # 2. Get and validate the reference payoff vector
        payoff_vectors = self.payoff_collection.get(ids=[tconst], include=['embeddings'])
        # --- FINAL FIX: Apply the same simplified check here ---
        if not payoff_vectors['ids']:
            print(f"Warning: Could not find payoff vector for tconst {tconst}")
            return pd.DataFrame()
        reference_payoff_vector = payoff_vectors['embeddings'][0]

        # 3. Find movies with a similar setup
        similar_setup_results = self.setup_collection.query(
            query_embeddings=[reference_setup_vector],
            n_results=50
        )
        
        candidate_ids = similar_setup_results['ids'][0]
        
        if tconst in candidate_ids:
            candidate_ids.remove(tconst)
        
        if not candidate_ids:
            return pd.DataFrame()

        # 4. Calculate the payoff dissimilarity for each candidate
        candidate_payoff_vectors = self.payoff_collection.get(ids=candidate_ids, include=['embeddings'])['embeddings']
        
        dissimilarities = []
        for candidate_vector in candidate_payoff_vectors:
            dist = cosine(reference_payoff_vector, candidate_vector)
            dissimilarities.append(dist)

        # 5. Rank candidates and return
        results_df = pd.DataFrame({
            'tconst': candidate_ids,
            'dissimilarity': dissimilarities
        })
        
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