from qa_with_faqs.rag import build_qa_system, ask_question  

def faq_retrieval(query: str) -> str:
    """
    Retrieve the most relevant documents from FAQ collection. 
    Use this tool when the user asks about F1 Racing.

    Input:
        query: str -> The user query to retrieve the most relevant documents
    """
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    
    searcher = build_qa_system()
    response = ask_question(searcher, query)
    return response