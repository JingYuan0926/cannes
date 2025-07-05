from uagents import Agent, Context, Model, Protocol
from uagents.experimental.quota import QuotaProtocol, RateLimit
from uagents_core.models import ErrorMessage
from enum import Enum
from datetime import datetime, UTC
from uuid import uuid4

from uagents_core.contrib.protocols.chat import (
    ChatMessage,
    ChatAcknowledgement,
    TextContent,
    chat_protocol_spec
)

chat_proto = Protocol(spec=chat_protocol_spec)

SEED_PHRASE = "shock table success cook desert note smart simple obey actress panic silent"

# Create agent with enhanced capabilities
agent = Agent(
    name="alice",
    port=8000,
    seed=SEED_PHRASE,
    mailbox=True,
    publish_agent_details=True,
    readme_path="README.md"
)

# Copy the address shown below
print(f"Your agent's address is: {agent.address}")

# Startup handler
@agent.on_event("startup")
async def startup_function(ctx: Context):
    ctx.logger.info(f"Hello, I'm agent {agent.name} and my address is {agent.address}.")
    ctx.logger.info("Alice is now ASI1 LLM compatible with chat protocol!")
    

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    for item in msg.content:
        if isinstance(item, TextContent):
            # Log received message
            ctx.logger.info(f"Received message from {sender}: {item.text}")
            
            # Send acknowledgment
            ack = ChatAcknowledgement(
                timestamp=datetime.now(UTC),
                acknowledged_msg_id=msg.msg_id
            )
            await ctx.send(sender, ack)
            
            # Send response message
            response_text = "Hello from Agent1!"
            response = ChatMessage(
                timestamp=datetime.now(UTC),
                msg_id=uuid4(),
                content=[TextContent(type="text", text=response_text)]
            )
            
            # Log the response message to console
            ctx.logger.info(f"Sending response to {sender}: {response_text}")
            
            await ctx.send(sender, response)

# Acknowledgement Handler - Process received acknowledgements
@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}")


agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    agent.run() 
    