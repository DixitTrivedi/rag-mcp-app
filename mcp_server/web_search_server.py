# Add project root to Python path
import sys
import os 
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# from fastmcp import FastMCP
from mcp.server.fastmcp import FastMCP
from tools.faq_retrieval_function import faq_retrieval
from tools.web_search_function import get_direct_answer
from dotenv import load_dotenv
import os

load_dotenv()

# Create an MCP server
mcp = FastMCP("MCP-RAG-app",
              host=os.getenv("HOST"),
              port=os.getenv("PORT"),
              stateless_http=True
              )

@mcp.tool()
def web_search_tool(query: str) -> str:
    """
    Search for information on a given topic using Bright Data.
    Use this tool when the user asks about a specific topic or question 
    that is not related to general machine learning.

    Input:
        query: str -> The user query to search for information

    Output:
        context: list[str] -> list of most relevant web search results
    """
    return get_direct_answer(query)