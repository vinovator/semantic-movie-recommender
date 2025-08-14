# 🎬 CineSearch: A Semantic Movie Recommender

CineSearch is an intelligent movie discovery engine built with Python and Streamlit. It goes beyond simple keyword matching, allowing you to find movies based on plot descriptions, thematic similarities, and a powerful combination of metadata filters.

---

## ✨ Key Features

* **Hybrid Search:** Combine semantic plot search with filters for actors, directors, genre, release year, and IMDb rating.
* **LLM-Powered Explanations:** Get a concise explanation from an LLM (powered by Google Gemini or a local Ollama model) on *why* a movie is a good match for your query.
* **Plot Twist Finder:** An advanced feature to discover movies that start with a similar premise but end in a completely different way.
* **Interactive UI:** A clean and user-friendly interface built with Streamlit, designed for easy movie discovery.

---

## 🛠️ Tech Stack

* **Backend:** Python
* **Data Manipulation:** Pandas, NumPy
* **Web Framework:** Streamlit
* **Embeddings:** Sentence Transformers
* **Vector Database:** ChromaDB
* **LLMs:** Google Gemini and/or Ollama (Llama 3)

---

## 🚀 Setup and Installation

Follow these steps to set up the project locally.

### 1. Prerequisites
* Python 3.9+
* An active internet connection for downloading models and data.
* (Optional) [Ollama](https://ollama.com/) installed and running if you wish to use a local LLM.

### 2. Clone the Repository

```bash
git clone [https://github.com/vinovator/semantic-movie-recommender.git](https://github.com/vinovator/semantic-movie-recommender.git)
cd semantic-movie-recommender
```

### 3. Set Up a Virtual Environment

It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 4. Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Download the Data

You need to download two datasets and place the files in the `data/raw/` directory.

  * **IMDb Datasets:** Download the following files from the [IMDb Non-Commercial Datasets](https://datasets.imdbws.com/) page:
      * `title.basics.tsv.gz`
      * `title.ratings.tsv.gz`
      * `name.basics.tsv.gz`
      * `title.principals.tsv.gz`
      * `title.crew.tsv.gz`
      * `title.akas.tsv.gz`
  * **Kaggle Wikipedia Plots:** Download the `wiki_movie_plots_deduped.csv` file from the [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) dataset on Kaggle.

### 6. Configure Environment Variables

Create a file named `.env` in the root of the project directory. If you plan to use the Google Gemini API, add your key to this file.

```
# .env file
GEMINI_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
```

-----

## 🏃‍♀️ Running the Application

Running the application for the first time involves a two-step data processing pipeline.

### Step 1: Process Raw Data

Run the first Jupyter Notebook to clean the raw data and create the master dataset.

  * Open and run all cells in `notebooks/01_data_exploration_and_cleaning.ipynb`.
  * This will generate the `movies_master_with_plots.parquet` file in the `data/processed/` directory.

### Step 2: Generate Embeddings & Build the Vector DB

Run the second Jupyter Notebook to create the vector embeddings and populate the ChromaDB database.

  * Open and run all cells in `notebooks/02_embedding_generation_and_indexing.ipynb`.
  * This will create the persistent vector database in the `db/` directory.

### Step 3: Launch the Streamlit App

Once the data pipeline has been run, you can start the web application.

  * Make sure your Ollama application is running if you have set `LLM_PROVIDER = 'ollama'` in `src/config.py`.
  * Run the following command from your terminal:
    ```bash
    streamlit run app.py
    ```

Your browser should automatically open to the CineSearch application interface.

-----

## 🔮 Future Improvements

  * Improve the data pipeline with fuzzy matching on titles to increase the number of movies with plot data.
  * Add more advanced search features, such as searching for movies based on a reference movie title.
  * Implement user accounts and a "watchlist" feature.
