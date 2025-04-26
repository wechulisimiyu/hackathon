from textwrap import dedent
from agno.agent import Agent
from agno.models.groq import Groq
from agno.playground import Playground, serve_playground_app
from agno.storage.sqlite import SqliteStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.lancedb import LanceDb
from agno.embedder.fastembed import FastEmbedEmbedder
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
import os

load_dotenv()

user_id = "Mkenya"

agent_storage: str = "tmp/agents.db"

# Initialize LanceDB
vector_db = LanceDb(
    table_name="blood_donation_knowledge",
    uri="tmp/lancedb",
    embedder=FastEmbedEmbedder(),
)

# Create PDF knowledge base 
knowledge_base = PDFKnowledgeBase(
    path="knowledge",
    vector_db=vector_db,
    reader=PDFReader(chunk=True)
)

# Load the knowledge base
knowledge_base.load(recreate=False)

memory = Memory(
    model=Groq(id="llama-3.3-70b-versatile"),
    db=SqliteMemoryDb(table_name="user_memories", db_file=agent_storage),
    delete_memories=True,  # Allow memory deletion
    clear_memories=True,   # Allow memory clearing
    memory_lifetime=30,    # Keep memories for 30 days
)

# Clear existing memories for fresh start
memory.clear()

donation_agent = Agent(
    name="Blood Donation Assistant",
    agent_id="donation_assistant",
    memory=memory,
    enable_agentic_memory=True,
    enable_user_memories=True,
    num_history_runs=5,
    read_chat_history=True,  # Enable chat history reading
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGoTools()],
    storage=SqliteStorage(table_name="agent_sessions", db_file=agent_storage),
    knowledge=knowledge_base,
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
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

# Add memory search function
def search_user_memories(user_id: str, query: str):
    memories = memory.search_user_memories(
        user_id=user_id,
        query=query,
        retrieval_method="agentic",
        limit=5
    )
    return memories

app = Playground(agents=[donation_agent]).get_app()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True, host="0.0.0.0", port=7777)