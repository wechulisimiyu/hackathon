from agno.agent import Agent
from agno.models.groq import Groq

agent = Agent(model=Groq(id="llama-3.3-70b-versatile"), markdown=True)
agent.print_response("What is the stock price of Apple?", stream=True)