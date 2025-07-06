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
import csv
import io
import asyncio
from dotenv import load_dotenv

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

# Analysis service configuration
ANALYSIS_SERVICE_URL = "http://localhost:3040/analyze"

# Startup handler
@agent.on_event("startup")
async def startup_function(ctx: Context):
    ctx.logger.info(f"Hello, I'm agent {agent.name} and my address is {agent.address}.")
    ctx.logger.info("Alice is now ready for data analysis using local analysis service!")

async def check_analysis_service_health() -> tuple[bool, str]:
    """Check if the analysis service is healthy"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:3040/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    return True, "Healthy"
                else:
                    return False, f"HTTP {response.status}"
    except Exception as e:
        return False, f"Error: {str(e)}"

async def convert_csv_to_json(csv_data: str) -> list:
    """Convert CSV data to JSON format for analysis"""
    try:
        # Parse CSV data
        reader = csv.DictReader(io.StringIO(csv_data))
        processed_data = []
        
        for row in reader:
            # Convert all values to strings to avoid type issues
            clean_row = {k: str(v) if v is not None else "" for k, v in row.items()}
            processed_data.append(clean_row)
        
        return processed_data
    except Exception as e:
        print(f"Error converting CSV to JSON: {e}")
        raise e

async def analyze_data_with_service(data: list, goal: str = "comprehensive data analysis") -> dict:
    """Send data to the analysis service for processing"""
    try:
        payload = {
            'data': data,
            'goal': goal
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                ANALYSIS_SERVICE_URL,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout for analysis
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Analysis service error {response.status}: {error_text}")
    except Exception as e:
        print(f"Error calling analysis service: {e}")
        raise e

def format_analysis_results(analysis_data: dict) -> str:
    """Format analysis results into readable text"""
    if not analysis_data:
        return "No analysis results available."
    
    formatted_text = "🤖 **AI DATA ANALYSIS RESULTS**\n\n"
    
    # Check if analysis was successful
    if analysis_data.get('status') == 'success':
        results = analysis_data.get('results', {})
        
        # Analysis summary
        summary = analysis_data.get('summary', {})
        if summary:
            formatted_text += "📊 **Analysis Summary:**\n"
            formatted_text += f"- Total Analyses: {summary.get('total_analyses', 0)}\n"
            formatted_text += f"- Total Graphs: {summary.get('total_graphs', 0)}\n"
            formatted_text += f"- Analytics Types: {', '.join(summary.get('analytics_types', []))}\n\n"
        
        # TEE Attestation
        tee_attestation = results.get('tee_attestation', {})
        if tee_attestation:
            formatted_text += "🔐 **TEE Attestation:**\n"
            if tee_attestation.get('tee_attested'):
                formatted_text += "✅ TEE Verified - Analysis executed in Trusted Execution Environment\n"
                formatted_text += f"- ROFL App ID: {tee_attestation.get('rofl_app_id', 'N/A')}\n"
                formatted_text += f"- Results Hash: {tee_attestation.get('results_hash', 'N/A')[:16]}...\n"
                formatted_text += f"- Algorithm: {tee_attestation.get('signature_algorithm', 'N/A')}\n"
                formatted_text += f"- Timestamp: {tee_attestation.get('timestamp', 'N/A')}\n"
            else:
                formatted_text += f"⚠️ TEE Unavailable: {tee_attestation.get('error', 'Unknown error')}\n"
            formatted_text += "\n"
        
        # Analysis Results
        analyses = results.get('analyses', [])
        if analyses:
            formatted_text += "**🔍 Analysis Results:**\n\n"
            for i, analysis in enumerate(analyses, 1):
                algorithm = analysis.get('algorithm', 'Unknown')
                analysis_type = analysis.get('analysis_type', 'Unknown')
                formatted_text += f"**{i}. {algorithm} - {analysis_type}**\n"
                
                # Analysis Insights
                if analysis.get('insights') and len(analysis['insights']) > 0:
                    formatted_text += "💡 **Key Insights:**\n"
                    for insight in analysis['insights']:
                        formatted_text += f"• {insight}\n"
                    formatted_text += "\n"
                
                # Additional information
                if analysis.get('description'):
                    formatted_text += f"**Description:** {analysis['description']}\n"
                
                if analysis.get('summary'):
                    formatted_text += f"**Summary:** {analysis['summary']}\n"
                
                # Performance metrics
                if analysis.get('performance'):
                    perf = analysis['performance']
                    formatted_text += "📈 **Performance Metrics:**\n"
                    for metric, value in perf.items():
                        if isinstance(value, float):
                            formatted_text += f"   • {metric}: {value:.4f}\n"
                        else:
                            formatted_text += f"   • {metric}: {value}\n"
                
                # Visualizations
                if analysis.get('graphs') and len(analysis['graphs']) > 0:
                    graph_count = len(analysis['graphs'])
                    formatted_text += f"📊 **Visualizations:** {graph_count} charts generated\n"
                    
                    # List chart titles if available
                    for graph in analysis['graphs']:
                        if graph.get('title'):
                            formatted_text += f"   • {graph['title']}\n"
                
                formatted_text += "\n"
        
        # Global insights
        if results.get('insights'):
            formatted_text += "🧠 **Global Insights:**\n"
            insights = results['insights']
            if isinstance(insights, list):
                for insight in insights:
                    formatted_text += f"• {insight}\n"
            elif isinstance(insights, str):
                formatted_text += f"• {insights}\n"
            formatted_text += "\n"
        
        # Analysis ID for reference
        if analysis_data.get('analysis_id'):
            formatted_text += f"🔖 **Analysis ID:** {analysis_data['analysis_id']}\n"
        
    else:
        # Error case
        error_msg = analysis_data.get('error', 'Unknown error occurred')
        formatted_text += f"❌ **Analysis Failed:** {error_msg}\n"
    
    return formatted_text

async def process_data_input(message_text: str) -> str:
    """Process user input and perform analysis"""
    try:
        # First check if analysis service is healthy
        is_healthy, health_status = await check_analysis_service_health()
        if not is_healthy:
            return f"❌ Analysis service is not available: {health_status}\n\nPlease ensure the analysis service is running on localhost:3040"
        
        data_to_analyze = []
        
        # Check if input is JSON
        if message_text.strip().startswith('{') or message_text.strip().startswith('['):
            try:
                json_data = json.loads(message_text)
                
                # Handle different JSON structures
                if isinstance(json_data, list):
                    data_to_analyze = json_data
                elif isinstance(json_data, dict):
                    if 'data' in json_data:
                        data_to_analyze = json_data['data']
                    elif 'processed_data' in json_data:
                        data_to_analyze = json_data['processed_data']
                    else:
                        # Treat the whole dict as a single data point
                        data_to_analyze = [json_data]
                else:
                    return "❌ Invalid JSON format. Expected array or object with 'data' field."
                
            except json.JSONDecodeError as e:
                return f"❌ Invalid JSON format: {str(e)}\n\nPlease ensure your JSON data is properly formatted."
        
        # Check if input looks like CSV
        elif ',' in message_text and '\n' in message_text:
            try:
                data_to_analyze = await convert_csv_to_json(message_text)
            except Exception as e:
                return f"❌ Error processing CSV data: {str(e)}\n\nPlease ensure your CSV data has headers and is properly formatted."
        
        else:
            return """❌ Unrecognized data format. Please provide data in one of these formats:

**JSON Array:**
```json
[
  {"name": "John", "age": 25, "city": "NYC"},
  {"name": "Jane", "age": 30, "city": "LA"}
]
```

**JSON Object with data field:**
```json
{
  "data": [
    {"name": "John", "age": 25, "city": "NYC"},
    {"name": "Jane", "age": 30, "city": "LA"}
  ]
}
```

**CSV Format:**
```
name,age,city
John,25,NYC
Jane,30,LA
```"""
        
        if not data_to_analyze:
            return "❌ No data found to analyze. Please provide valid data."
        
        # Perform analysis
        analysis_result = await analyze_data_with_service(data_to_analyze, "comprehensive data analysis")
        
        # Format and return results
        return format_analysis_results(analysis_result)
        
    except Exception as e:
        return f"❌ Analysis failed: {str(e)}\n\nPlease ensure:\n1. Your data is in the correct format\n2. The analysis service is running on localhost:3040\n3. The data contains valid columns and rows"

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    for item in msg.content:
        if isinstance(item, TextContent):
            # Log received message
            ctx.logger.info(f"Received message from {sender}")
            
            # Send acknowledgment
            ack = ChatAcknowledgement(
                timestamp=datetime.now(UTC),
                acknowledged_msg_id=msg.msg_id
            )
            await ctx.send(sender, ack)
            
            # Check if message contains data
            message_text = item.text.strip()
            
            # Check if message looks like data (JSON or CSV)
            if (len(message_text) > 50 and 
                (message_text.startswith('{') or message_text.startswith('[') or 
                 (',' in message_text and '\n' in message_text))):
                
                ctx.logger.info("Processing data for analysis...")
                response_text = await process_data_input(message_text)
                
            else:
                # Regular message - provide instructions
                response_text = """🤖 **AI Data Analysis Agent**

I can analyze your data using advanced AI and machine learning algorithms!

**How to use:**
1. Paste your data directly in this chat (JSON or CSV format)
2. I'll automatically detect the format and run comprehensive analysis
3. Get detailed insights, visualizations, and TEE-verified results

**Supported formats:**

**JSON Array:**
```json
[
  {"name": "John", "age": 25, "salary": 50000},
  {"name": "Jane", "age": 30, "salary": 60000}
]
```

**JSON Object:**
```json
{
  "data": [
    {"name": "John", "age": 25, "salary": 50000},
    {"name": "Jane", "age": 30, "salary": 60000}
  ]
}
```

**CSV Format:**
```
name,age,salary
John,25,50000
Jane,30,60000
```

**What I analyze:**
• 📊 Descriptive Analytics (Clustering, Dimensionality Reduction)
• 🔮 Predictive Analytics (Regression, Classification)
• 💡 Prescriptive Analytics (Optimization, Recommendations)
• 🔍 Diagnostic Analytics (Feature Importance, Causal Analysis)
• 📈 Statistical Analysis & Visualizations
• 🔐 TEE-Verified Results

**Analysis Pipeline:**
1. 🔄 Data Processing & Type Conversion
2. 🧠 AI-Powered Algorithm Selection
3. 📊 Multi-Modal Analysis Execution
4. 📈 Visualization Generation
5. 🔐 TEE Attestation & Signing
6. 📝 Comprehensive Insights Report

Just paste your data and I'll handle the rest! 🚀

*Note: Analysis service must be running on localhost:3040*"""
            
            # Send response message
            response = ChatMessage(
                timestamp=datetime.now(UTC),
                msg_id=uuid4(),
                content=[TextContent(type="text", text=response_text)]
            )
            
            # Log the response
            ctx.logger.info(f"Sending response to {sender}")
            
            await ctx.send(sender, response)

@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}")

agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    agent.run() 
    