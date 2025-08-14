# src/llm_handler.py

import ollama
import google.generativeai as genai
import yaml
from . import config

# Load prompts from yaml file
try:
    with open("prompts.yaml", "r") as f:
        PROMPTS = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: prompts.yaml not found. Make sure it exists in project root folder")
    PROMPTS = {}

def get_explanation(recommended_movie, query_text):
    """
    Generates an explanation for why a movie is recommended based on a query text.
    """

    if not PROMPTS:
        return "Error: Could not load prompts from prompts.yaml."
    
    prompt_template = PROMPTS.get("explain_similarity_from_text", {}).get("template")
    if not prompt_template:
        return "Error: 'explain_similarity_from_text' prompt not found in prompts.yaml."

    user_prompt = prompt_template.format(
        query_description=query_text,
        recommendation_title=recommended_movie['primaryTitle'],
        recommendation_plot=recommended_movie['plot']
    )

    try:
        if config.LLM_PROVIDER == "gemini":
            genai.configure(api_key = config.GEMINI_API_KEY)
            model = genai.GenerativeModel(config.GEMINI_MODEL)
            response = model.generate_content(user_prompt)
            return response.text.strip()
        
        elif config.LLM_PROVIDER == "ollama":
            response = ollama.chat(
               model=config.OLLAMA_MODEL,
                messages=[
                    # The system prompt is now part of the template, so we just send the user prompt
                    {'role': 'user', 'content': user_prompt},
                ]
            )
            return response['message']['content']
        
        else:
            return "Error: Unsupported LLM provider specified in config.py."
        
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return "Error generating explanation. Please try again later."
    

if __name__ == "__main__":
    # Example usage
    query_movie = {
        'primaryTitle': 'Inception',
        'plot': 'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO.'
    }
    
    recommended_movie = {
        'primaryTitle': 'Interstellar',
        'plot': 'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.'
    }
    
    explanation = get_explantaion(query_movie, recommended_movie)
    print(explanation)