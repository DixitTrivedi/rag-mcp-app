from serpapi import GoogleSearch
import os
from dotenv import load_dotenv

load_dotenv()

import os
from serpapi import GoogleSearch

def get_direct_answer(query):
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY"),
    }

    search = GoogleSearch(params)
    results = search.get_dict()


    # Try to get answer from different sources
    if 'answer_box' in results and 'answer' in results['answer_box']:
        return results['answer_box']['answer']
    elif 'answer_box' in results and 'snippet' in results['answer_box']:
        return results['answer_box']['snippet']
    elif 'knowledge_graph' in results and 'description' in results['knowledge_graph']:
        return results['knowledge_graph']['description']
    elif 'organic_results' in results:
        return results['organic_results'][0]['snippet']
    else:
        return "No direct answer found."

# Usage:
# print(get_direct_answer("What is the capital of France?"))


# Example usage
# if __name__ == "__main__":
#     query = "What is machine learning?"
#     results = get_direct_answer(query)
#     print(results)