import os 
import json 
from src.cache_utils.semantic_cache_response import SemanticCache
from src.agents_utils.agents import judicial_agent
from dotenv import load_dotenv

def get_response(query:str, chat_history:list, section_type="BNS"):
    
    config_file = "./artifacts/config.json"
    with open(config_file, "r") as file:
        config = json.load(file)
    
    cache_collection = config["CACHE_COLLECTION"]["semantic-cache"]
    
    embedding_model = config["EMBEDDING_MODEL"]

    llm = config["LLAMA"]
    
    """Cache responses"""
    semantic_cache = SemanticCache(embedding_model, cache_collection,
                                   llm)
    
    # Check if the anwer present in the cache
    cached_response = semantic_cache.find_in_cache(query)
    if cached_response['output']:
        return cached_response

    # If the answer is not in the cache, generate it using the judicial agent
    response = judicial_agent(query, section_type, chat_history)
    
    # Add the response to the cache
    semantic_cache.add_to_cache(query, response)
    
    return response

    