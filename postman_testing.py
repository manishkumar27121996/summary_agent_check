import asyncio,sys
from os import getenv
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agno.agent import Agent
from agno.tools.mcp import MCPTools
from agno.models.aws import Claude
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None

# Global agent variable
agent = None
mcp_tools = None

async def initialize_agent():
    """Initialize the agent with MCP tools"""
    global agent, mcp_tools
    if agent is None:
        mcp_tools = MCPTools(
            "npx -y mongodb-mcp-server --connectionString 'mongodb+srv://krmaan20010:manish123@cluster0.nolvbco.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'",
            timeout_seconds=90
        )
        await mcp_tools.__aenter__()
        
        agent = Agent(
            model=Claude(id="us.anthropic.claude-3-5-sonnet-20241022-v2:0"),
            tools=[mcp_tools],
            markdown=True,
            add_history_to_messages=True,
            description="You are a MongoDB summarization assistant that queries and interprets worklogs from the 'summary' collection inside 'summarization_collection'.",
            instructions=[
                "Always respond with concise sentences.",
                "Wrap responses in markdown code blocks when returning data.",
                (
                    "You are connected to a MongoDB instance. "
                    "Always query the `summary` collection inside the `summarization_collection` database unless explicitly told otherwise.\n\n"
                    "**How to Handle Queries:**\n"
                    "- For **person-based queries**, use case-insensitive regex matching on the `name` field.\n"
                    "- For **project-related queries**, match on the `project_name` field.\n"
                    "- Use `timesheet_date` to find the **most recent worklog**, sorting in descending order.\n"
                    "- If asked to **summarize** someone's worklog, extract and briefly describe:\n"
                    "  - The `title`\n"
                    "  - Applications used (`applications` field)\n"
                    "  - Role, department, project name, and tools\n"
                    "  - Total billed time (`logged_duration`) and cost (`total_billed_cost`)\n"
                    "- If asked for **team members**, aggregate unique names and emails from the `name` and `email` fields.\n"
                    "- If asked for someone's **latest update**, return their latest entry based on `timesheet_date`.\n"
                    "- If a name or project doesn't return results, respond politely with a clarification request.\n\n"
                    "**Data Schema Notes:**\n"
                    "- `name`, `email`, `role`, `designation_name`, `department_name`, `company_name`, `project_name`, `timesheet_date`\n"
                    "- `logged_duration` is in seconds; convert to hours/minutes as needed.\n"
                    "- Use regex for names like 'Akshay', 'Vishal', etc., in case full names aren't provided.\n"
                    "- Avoid accessing other collections unless explicitly instructed."
                )
            ]
        )

async def cleanup_agent():
    """Cleanup the agent and MCP tools"""
    global mcp_tools
    if mcp_tools:
        try:
            await mcp_tools.__aexit__(None, None, None)
        except Exception as e:
            print(f"Error during cleanup: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    print("Starting up MongoDB Chatbot API...")
    await initialize_agent()
    print("Agent initialized successfully!")
    
    yield
    
    # Shutdown
    print("Shutting down MongoDB Chatbot API...")
    await cleanup_agent()
    print("Cleanup completed!")

# Create FastAPI app with lifespan
app = FastAPI(
    title="MongoDB Chatbot API",
    description="A MongoDB summarization assistant that queries and interprets worklogs",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat with the MongoDB assistant
    """
    try:
        if agent is None:
            await initialize_agent()
        
        response = await agent.arun(request.message)
        
        return ChatResponse(
            response=response.content,
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "MongoDB Chatbot API is running"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MongoDB Chatbot API",
        "description": "A MongoDB summarization assistant that queries and interprets worklogs",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    # Set your Anthropic API key: export ANTHROPIC_API_KEY=***
    uvicorn.run("postman_testing:app", host="0.0.0.0", port=8005)
