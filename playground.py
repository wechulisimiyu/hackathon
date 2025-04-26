from textwrap import dedent
from agno.agent import Agent
from agno.models.groq import Groq
from agno.playground import Playground, serve_playground_app
from agno.storage.sqlite import SqliteStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

agent_storage: str = "tmp/agents.db"

donation_agent = Agent(
    name="Blood Donation Assistant",
    agent_id="donation_assistant",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGoTools()],
    storage=SqliteStorage(table_name="agent_sessions", db_file=agent_storage),
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