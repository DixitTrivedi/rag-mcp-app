# -------------------------------------------------
#  With FastAPI
# -------------------------------------------------

import contextlib
from fastapi import FastAPI
from mcp_server.rag_server import mcp as rag_mcp
from mcp_server.web_search_server import mcp as web_search_mcp
import os


# Create a combined lifespan to manage both session managers
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(rag_mcp.session_manager.run())
        await stack.enter_async_context(web_search_mcp.session_manager.run())
        yield


app = FastAPI(lifespan=lifespan)
app.mount("/rag", rag_mcp.streamable_http_app())
app.mount("/web_search", web_search_mcp.streamable_http_app())

PORT = os.environ.get("PORT", 10000)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


# -------------------------------------------------
#  Test Server in MCP Inspector
# -------------------------------------------------

# -------------------------------------------------
#  Set Streamable HTTP 

# -------------------------------------------------
# http://0.0.0.0:8080/web_search/mcp


# -------------------------------------------------
# http://0.0.0.0:8080/rag/mcp