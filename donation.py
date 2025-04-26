from textwrap import dedent
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from agno.agent import Agent
from agno.playground import Playground, serve_playground_app
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.storage.sqlite import SqliteStorage
import json
import asyncio

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only. In production, specify your UI domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_donation_assistant(model_id: str = "llama-3.3-70b-versatile") -> Agent:
    # Create a storage backend using SQLite
    storage = SqliteStorage(
        table_name="agent_sessions",
        db_file="tmp/data.db"
    )

    return Agent(
        name="DonationAssistant",
        agent_id="donation_assistant",
        model=Groq(id=model_id),
        tools=[DuckDuckGoTools()],
        storage=storage,
        description=dedent("""\
            You are a Blood Donation Assistant, specialized in providing accurate information about blood donation 
            and addressing common myths and misconceptions. Your goal is to educate people about blood donation,
            encourage safe donation practices, and help save lives through informed blood donation.

            Key responsibilities:
            1. Provide accurate medical information about blood donation
            2. Address and debunk common myths with scientific evidence
            3. Explain eligibility criteria and preparation steps
            4. Guide potential donors through the donation process
            5. Emphasize the importance and impact of blood donation
            6. Direct users to appropriate medical resources when needed"""),
        markdown=True,
    )

agent = get_donation_assistant()

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if "message" not in data:
                await websocket.send_json({"error": "No message provided"})
                continue
                
            # Get response from agent
            async for chunk in agent.astream_response(data["message"]):
                await websocket.send_json({
                    "type": "chunk",
                    "content": chunk
                })
            
            # Send completion message
            await websocket.send_json({
                "type": "done"
            })
            
    except Exception as e:
        await websocket.close()

@app.get("/")
async def root():
    return {"status": "ok", "message": "Blood Donation Assistant API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("donation:app", host="127.0.0.1", port=8000, reload=True)