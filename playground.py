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
from datetime import datetime, timedelta
from resend import Emails
import ics
from urllib.parse import quote

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
)

# Clear existing memories for fresh start
memory.clear()

# Initialize Resend API key
Emails.api_key = os.environ["RESEND_API_KEY"]

def calculate_next_donation_date(gender: str, last_donation: datetime) -> datetime:
    interval = timedelta(days=120) if gender.lower() == 'male' else timedelta(days=180)
    return last_donation + interval

def create_google_calendar_link(event_name: str, start_date: datetime, description: str, location: str = "Blood Donation Center"):
    date_str = start_date.strftime("%Y%m%dT%H%M%S")
    end_date = (start_date + timedelta(hours=1)).strftime("%Y%m%dT%H%M%S")
    base_url = "https://calendar.google.com/calendar/render"
    params = {
        "action": "TEMPLATE",
        "text": quote(event_name),
        "dates": f"{date_str}/{end_date}",
        "details": quote(description),
        "location": quote(location)
    }
    return f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"

def create_ics_file(event_name: str, start_date: datetime, description: str, location: str = "Blood Donation Center"):
    calendar = ics.Calendar()
    event = ics.Event()
    event.name = event_name
    event.begin = start_date
    event.end = start_date + timedelta(hours=1)
    event.description = description
    event.location = location
    calendar.events.add(event)
    return calendar.serialize()

def send_donation_reminder(email: str, appointment_date: datetime):
    event_name = "Blood Donation Appointment"
    description = """
    Please remember:
    - Get adequate rest
    - Eat a healthy meal
    - Stay hydrated
    - Bring valid ID
    """
    
    # Create calendar links
    google_calendar_link = create_google_calendar_link(event_name, appointment_date, description)
    ics_content = create_ics_file(event_name, appointment_date, description)
    
    params: Emails.SendParams = {
        "from": os.getenv("DONATION_EMAIL_FROM", "blood-donation@yourdomain.com"),
        "to": [email],
        "subject": "Blood Donation Appointment Reminder",
        "html": f"""
        <h1>Your Blood Donation Appointment</h1>
        <p>Your next donation appointment is scheduled for: {appointment_date.strftime('%B %d, %Y at %I:%M %p')}</p>
        <p>Please remember to:</p>
        <ul>
            <li>Get adequate rest</li>
            <li>Eat a healthy meal</li>
            <li>Stay hydrated</li>
            <li>Bring valid ID</li>
        </ul>
        <p>
            <a href="{google_calendar_link}" style="display: inline-block; padding: 10px 20px; background-color: #4285f4; color: white; text-decoration: none; border-radius: 5px; margin-right: 10px;">
                Add to Google Calendar
            </a>
        </p>
        """,
        "attachments": [
            {
                "filename": "blood_donation_appointment.ics",
                "content": ics_content
            }
        ]
    }
    
    email_result = Emails.send(params)
    return email_result

# Information Assistant
info_agent = Agent(
    name="Blood Donation Information Assistant",
    agent_id="info_assistant",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGoTools()],
    storage=SqliteStorage(table_name="info_sessions", db_file=agent_storage),
    knowledge=knowledge_base,
    description="You are a Blood Donation Information Specialist, focused on education and answering questions.",
    instructions=dedent("""\
        Your role is to provide accurate information about blood donation:
        1. Answer questions about eligibility, process, and safety
        2. Address myths and misconceptions with scientific evidence
        3. Provide detailed explanations about donation process
        4. Direct users to proper medical resources when needed
        
        Always:
        - Cite reputable sources
        - Include medical disclaimers
        - Keep responses clear and simple
        - Direct medical questions to healthcare providers"""),
    add_datetime_to_instructions=True,
    markdown=True
)

# Scheduling Assistant
scheduling_agent = Agent(
    name="Blood Donation Scheduling Assistant",
    agent_id="scheduling_assistant",
    model=Groq(id="llama-3.3-70b-versatile"),
    memory=memory,
    storage=SqliteStorage(table_name="scheduling_sessions", db_file=agent_storage),
    description="You are a Blood Donation Scheduling Specialist, focused on managing appointments and reminders.",
    instructions=dedent("""\
        Your role is to handle donation scheduling:
        
        First Interaction:
        1. Ask for gender (separate question)
        2. After gender is provided, ask for email
        3. Store both in memory
        
        For Scheduling:
        1. Calculate next eligible date based on gender
        2. Send confirmation email with calendar integration
        3. Track donation history
        
        Always:
        - Keep questions separate (gender first, then email)
        - Verify all required information before scheduling
        - Send clear confirmation emails
        - Track intervals (4 months male, 6 months female)"""),
    add_datetime_to_instructions=True,
    markdown=True
)

# Agent Tools
class AgentTools:
    def __init__(self, info_agent, scheduling_agent):
        self.info_agent = info_agent
        self.scheduling_agent = scheduling_agent

    def route_to_info(self, query: str) -> str:
        return self.info_agent.get_response(query)

    def route_to_scheduler(self, query: str) -> str:
        return self.scheduling_agent.get_response(query)

agent_tools = AgentTools(info_agent, scheduling_agent)

# Coordinator Agent
coordinator_agent = Agent(
    name="Blood Donation Coordinator",
    agent_id="coordinator",
    model=Groq(id="llama-3.3-70b-versatile"),
    storage=SqliteStorage(table_name="coordinator_sessions", db_file=agent_storage),
    tools=[agent_tools],
    description="You are a Blood Donation Coordinator who routes requests to specialized agents.",
    instructions=dedent("""\
        Your role is to:
        1. Determine user intent and route to appropriate specialist
        2. For information queries: Use route_to_info()
        3. For scheduling requests: Use route_to_scheduler()
        
        Classification Rules:
        - Scheduling: mentions of appointment, booking, dates, availability
        - Information: questions about process, eligibility, safety
        
        Always:
        - Analyze user intent first
        - Route to appropriate specialist using the correct tool
        - Maintain conversation context
        - Be clear about which specialist is handling the request"""),
    add_datetime_to_instructions=True,
    markdown=True
)

# Initialize agents
info_agent.initialize_agent()
scheduling_agent.initialize_agent()
coordinator_agent.initialize_agent()

# Create playground with coordinator
app = Playground(agents=[coordinator_agent]).get_app()

# Add memory search function
def search_user_memories(user_id: str, query: str):
    memories = memory.search_user_memories(
        user_id=user_id,
        query=query,
        retrieval_method="agentic",
        limit=5
    )
    return memories

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