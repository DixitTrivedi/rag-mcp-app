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
def faq_retrieval_tool(query: str) -> str:
    """Retrieve the most relevant documents from FAQ collection. 
    Use this tool when the user asks about F1 Racing.

    Input:
        query: str -> The user query to retrieve the most relevant documents

    Output:
        context: str -> most relevant documents retrieved from a vector DB
    """
    return faq_retrieval(query)