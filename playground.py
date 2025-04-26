from textwrap import dedent
from agno.agent import Agent
from agno.team import Team
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
# memory.clear()

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
    description="You are a Blood Donation Scheduling Specialist",
    instructions=dedent("""\
        Follow these steps in order:

        Step 1: Check memory for gender
        - If missing, ask and store using memory.add(key='gender')
        - If present, move to Step 2

        Step 2: Check memory for email
        - If missing, ask and store using memory.add(key='email') 
        - If present, move to Step 3

        Step 3: Begin Scheduling
        - Ask for preferred date and time
        - Check eligibility based on gender interval
        - Send confirmation or suggest next available date

        Important:
        - Never ask for information you already have
        - Move to next step immediately after storing information
        - Keep communication friendly and clear"""),
    add_datetime_to_instructions=True,
    markdown=True
)
# Create a team with route mode
donation_team = Team(
    name="Blood Donation Team",
    mode="route",  # Set to route mode
    model=Groq(id="llama-3.3-70b-versatile"),
    members=[
        info_agent,
        scheduling_agent
    ],
    show_tool_calls=True,
    markdown=True,
    instructions=[
        "You are a Blood Donation Team Coordinator that routes queries to specialized agents.",
        "Route information/eligibility questions to the Information Assistant.",
        "Route scheduling/appointment requests to the Scheduling Assistant.",
        "Classification Rules:",
        "- Information Assistant: Questions about process, eligibility, safety, requirements",
        "- Scheduling Assistant: Mentions of booking, appointments, dates, scheduling",
        "Always analyze the intent before routing to ensure accurate agent selection."
    ],
    show_members_responses=True
)

# Initialize the team
donation_team.initialize_team()

# Create playground with team instead of coordinator agent
app = Playground(agents=[info_agent, scheduling_agent]).get_app()

# ...rest of the code remains the same...

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