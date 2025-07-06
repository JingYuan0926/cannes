from uagents import Agent, Context, Model, Protocol
from uagents.experimental.quota import QuotaProtocol, RateLimit
from uagents_core.models import ErrorMessage
from enum import Enum
from datetime import datetime, UTC
from uuid import uuid4
import aiohttp
import json
import os
import hashlib
from dotenv import load_dotenv
import asyncio

from uagents_core.contrib.protocols.chat import (
    ChatMessage,
    ChatAcknowledgement,
    TextContent,
    chat_protocol_spec
)

# Load environment variables
load_dotenv()

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
    

async def get_attestation_info(model_name: str):
    """Get attestation information for the specified model"""
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.red-pill.ai/v1/attestation/report?model={model_name}"
            async with session.get(url) as response:
                if response.status == 200:
                    attestation_data = await response.json()
                    return attestation_data.get("signing_address", ""), attestation_data
                else:
                    return "", None
    except Exception as e:
        print(f"Error getting attestation: {e}")
        return "", None

async def get_signature_info(request_id: str, model_name: str, api_key: str):
    """Get signature information for verification"""
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            url = f"https://api.red-pill.ai/v1/signature/{request_id}?model={model_name}&signing_algo=ecdsa"
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    try:
                        # Try to parse as JSON first
                        return await response.json()
                    except:
                        # If JSON parsing fails, try to parse the text as JSON
                        text_response = await response.text()
                        try:
                            return json.loads(text_response)
                        except:
                            print(f"Could not parse signature response: {text_response}")
                            return None
                else:
                    error_text = await response.text()
                    print(f"Signature API error {response.status}: {error_text}")
                    return None
    except Exception as e:
        print(f"Error getting signature: {e}")
        return None

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
            
            # Get attestation info first
            model_name = "phala/llama-3.3-70b-instruct"
            signing_address, attestation_data = await get_attestation_info(model_name)
            
            # Call RedPill AI API to get response
            try:
                api_key = os.getenv('REDPILL_API_KEY')
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}"
                    }
                    
                    payload = {
                        "model": model_name,
                        "messages": [
                            {
                                "role": "user",
                                "content": item.text
                            }
                        ]
                    }
                    
                    # Calculate request body hash for verification
                    request_body = json.dumps(payload, separators=(',', ':'))
                    request_hash = hashlib.sha256(request_body.encode()).hexdigest()
                    
                    async with session.post(
                        "https://api.red-pill.ai/v1/chat/completions",
                        headers=headers,
                        json=payload
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            response_content = result["choices"][0]["message"]["content"]
                            request_id = result.get("id", "")
                            
                            # Calculate response hash
                            response_body = await response.text()
                            response_hash = hashlib.sha256(response_body.encode()).hexdigest()
                            
                            # Get signature information (wait a bit for it to be available)
                            await asyncio.sleep(2)  # Wait 2 seconds for signature to be processed
                            signature_info = await get_signature_info(request_id, model_name, api_key)
                            
                            # Format response with TEE verification info
                            if signature_info:
                                response_text = f"{response_content}\n\n--- TEE Verification ---\nSigning Address:\n{signature_info.get('signing_address', 'N/A')}\nMessage Hash:\n{signature_info.get('text', 'N/A')}\nSignature Hash:\n{signature_info.get('signature', 'N/A')}"
                                
                                ctx.logger.info(f"TEE Verification successful:")
                                ctx.logger.info(f"  Signing Address: {signature_info.get('signing_address', 'N/A')}")
                                ctx.logger.info(f"  Message Hash: {signature_info.get('text', 'N/A')}")
                                ctx.logger.info(f"  Signature Hash: {signature_info.get('signature', 'N/A')}")
                            else:
                                response_text = f"{response_content}\n\n--- TEE Verification ---\nSigning Address:\n{signing_address or 'N/A'}\nMessage Hash:\nN/A\nSignature Hash:\nN/A"
                                
                        else:
                            ctx.logger.error(f"API call failed with status {response.status}")
                            response_text = "Sorry, I'm having trouble processing your request right now."
                            
            except Exception as e:
                ctx.logger.error(f"Error calling RedPill AI API: {e}")
                response_text = "Sorry, I encountered an error while processing your request."
            
            # Send response message
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
    