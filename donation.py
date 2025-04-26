from textwrap import dedent
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.storage.sqlite import SqliteStorage

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

if __name__ == "__main__":
    assistant = get_donation_assistant()
    while True:
        question = input("\nAsk a question about blood donation (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        assistant.print_response(question, stream=True)