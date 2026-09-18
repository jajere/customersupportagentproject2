"""
Customer Support AI Agent — Production Code
==========================================
Run locally:
    uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
    agentcore deploy

Invoke deployed agent:
    agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import argparse
import asyncio
import json
import logging
import os
import uuid
from typing import Dict

import boto3
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.tools.code_interpreter_client import code_session
from mcp.client.streamable_http import streamable_http_client
from strands import Agent, tool
from strands.hooks import (
    AfterInvocationEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from strands_tools.browser import AgentCoreBrowser

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── 1. App Initialisation ─────────────────────────────────────────────────────
app = BedrockAgentCoreApp()

# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"

# ── 2. Configuration ──────────────────────────────────────────────────────────
GATEWAY_URL = "https://customersupportgateway-xsidpyyy0b.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID = "VIQGLRSDNZ"
REGION = "us-east-1"
MEMORY_ID = "CustomerSupportMemory-eLFp2pGr8x"

# ── 3. Model and Clients ──────────────────────────────────────────────────────
model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id, region_name=REGION)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── 4. Namespace Helper ───────────────────────────────────────────────────────
def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string."""
    try:
        strategies = mem_client.get_memory_strategies(memory_id)
        return {
            strategy["type"]: strategy["namespaces"][0]
            for strategy in strategies
            if "type" in strategy and "namespaces" in strategy and strategy["namespaces"]
        }
    except Exception as e:
        logger.warning(f"Failed to fetch memory strategy namespaces: {e}")
        return {
            "SEMANTIC": "cs_agent/{actorId}/facts",
            "USER_PREFERENCE": "cs_agent/{actorId}/preferences"
        }


# ── 5. Memory Hook ────────────────────────────────────────────────────────────
class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(self.memory_client, self.memory_id)

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        try:
            messages = event.agent.messages
            if not messages:
                return

            last_msg = messages[-1]
            if getattr(last_msg, "role", "") != "user":
                return

            # Extract plain query text
            query_text = ""
            if isinstance(last_msg.content, str):
                query_text = last_msg.content
            elif isinstance(last_msg.content, list):
                text_parts = [c.get("text", "") for c in last_msg.content if isinstance(c, dict) and "text" in c]
                query_text = " ".join(text_parts)

            if not query_text or query_text.startswith("Customer Context:"):
                return

            collected_memories = []
            for strat_type, ns_template in self.namespaces.items():
                formatted_ns = ns_template.format(actorId=self.actor_id)
                memories = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=formatted_ns,
                    query=query_text,
                    top_k=5
                )
                for mem in memories:
                    mem_text = mem.get("text") or mem.get("content")
                    if mem_text:
                        collected_memories.append(f"[{strat_type}] {mem_text}")

            if collected_memories:
                context_str = "\n".join(collected_memories)
                new_content = f"Customer Context:\n{context_str}\n\n{query_text}"
                last_msg.content = new_content

        except Exception as e:
            logger.warning(f"Error in retrieve_customer_context: {e}")

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        try:
            messages = event.agent.messages
            if not messages:
                return

            last_user_query = None
            last_assistant_resp = None

            for msg in reversed(messages):
                role = getattr(msg, "role", None)
                if not last_assistant_resp and role == "assistant":
                    if isinstance(msg.content, str):
                        last_assistant_resp = msg.content
                    elif isinstance(msg.content, list):
                        parts = [c.get("text", "") for c in msg.content if isinstance(c, dict) and "text" in c]
                        last_assistant_resp = "\n".join(parts)
                elif not last_user_query and role == "user":
                    if isinstance(msg.content, str):
                        last_user_query = msg.content
                    elif isinstance(msg.content, list):
                        parts = [c.get("text", "") for c in msg.content if isinstance(c, dict) and "text" in c]
                        last_user_query = "\n".join(parts)

                if last_user_query and last_assistant_resp:
                    break

            if last_user_query and last_assistant_resp:
                # Remove memory header prefix if it was prepended
                if "Customer Context:\n" in last_user_query and "\n\n" in last_user_query:
                    last_user_query = last_user_query.split("\n\n", 1)[1]

                self.memory_client.create_event(
                    memory_id=self.memory_id,
                    actor_id=self.actor_id,
                    session_id=self.session_id,
                    messages=[
                        (last_user_query, "USER"),
                        (last_assistant_resp, "ASSISTANT")
                    ]
                )
        except Exception as e:
            logger.warning(f"Error in save_support_interaction: {e}")

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register both memory callbacks."""
        registry.register(MessageAddedEvent, self.retrieve_customer_context)
        registry.register(AfterInvocationEvent, self.save_support_interaction)


# ── 6. Knowledge Base Tool ────────────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID:
        return "Knowledge base not configured."

    try:
        resp = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query}
        )
        results = resp.get("retrievalResults", [])
        if not results:
            return "No relevant information found in the knowledge base."

        chunks = [res["content"]["text"] for res in results if "content" in res and "text" in res["content"]]
        return "\n---\n".join(chunks) if chunks else "No relevant content found."
    except Exception as e:
        return f"Error querying Knowledge Base: {str(e)}"


# ── 7. Loyalty Discount Tool (Code Interpreter) ───────────────────────────────
@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    code = f"""
import json, math

points = {loyalty_points}
tier = "{tier}".capitalize()
total = {order_total}
category = "{product_category}".lower()

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

# Calculate points redemption (floor to nearest 500, cap at 50% of order value; 100 pts = $1)
max_points_allowed = int((total * 0.5) * 100)
redeemable_points = min(points, max_points_allowed)
points_redeemed = (redeemable_points // 500) * 500
points_discount = points_redeemed / 100.0

subtotal_after_points = max(0.0, total - points_discount)
tier_rate = tier_rates.get(tier, 0.0)
tier_discount = round(subtotal_after_points * tier_rate, 2)

final_total = max(0.0, subtotal_after_points - tier_discount)
total_savings = round(points_discount + tier_discount, 2)

earn_rate = earn_rates.get(category, 1)
points_earned = int(final_total * earn_rate)
remaining_points = points - points_redeemed + points_earned

result = {{
    "original_total": round(total, 2),
    "tier": tier,
    "product_category": category,
    "points_redeemed": points_redeemed,
    "points_discount_usd": round(points_discount, 2),
    "tier_discount_usd": tier_discount,
    "total_savings_usd": total_savings,
    "final_total_usd": round(final_total, 2),
    "points_earned": points_earned,
    "remaining_points": remaining_points
}}

print(json.dumps(result))
"""

    try:
        with code_session(REGION) as session:
            response = session.invoke("executeCode", {
                "code": code,
                "language": "python",
                "clearContext": True
            })

            for event in response.get("stream", []):
                if "result" in event:
                    return event["result"].get("stdout", "").strip()
                elif "stdout" in event:
                    return event["stdout"].strip()

            return json.dumps({"status": "executed", "output": "Code executed with no stdout."})

    except Exception as e:
        tier_map = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        t_rate = tier_map.get(tier.capitalize(), 0.0)
        t_discount = round(order_total * t_rate, 2)
        f_total = max(0.0, order_total - t_discount)

        return json.dumps({
            "original_total": round(order_total, 2),
            "tier": tier,
            "tier_discount_usd": t_discount,
            "final_total_usd": round(f_total, 2),
            "fallback_note": f"Code interpreter unavailable ({str(e)}). Applied basic tier discount."
        })


# ── 8. Agent Entrypoint ───────────────────────────────────────────────────────
@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    try:
        user_input = payload.get("prompt", "")
        actor_id = payload.get("customer_id", "CUST-ANONYMOUS")
        session_id = payload.get("session_id") or str(uuid.uuid4())

        # 1. Instantiate Memory Hook
        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID
        )

        # 2. Instantiate AgentCore Browser
        agent_core_browser = AgentCoreBrowser(region=REGION)

        # 3. Base Tools
        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            agent_core_browser.browser
        ]

        # 4. Connect to Gateway MCP Client & retrieve tools
        async with streamable_http_client(GATEWAY_URL) as (read_stream, write_stream):
            async with MCPClient(read_stream, write_stream) as mcp_client:
                gateway_tools = await mcp_client.get_tools()
                tools.extend(gateway_tools)

                system_prompt = (
                    "You are an intelligent, empathetic customer support AI agent for an Amazon online store. "
                    "You assist customers with order status, returns/refunds, product specs, store policies, "
                    "loyalty calculations, and website navigation. Be concise, helpful, and polite."
                )

                # 5. Create Agent with fixed memory_hook reference
                agent = Agent(
                    model=model,
                    tools=tools,
                    hooks=[memory_hook],
                    system_prompt=system_prompt
                )

                # 6. Invoke Agent while MCP stream context is active
                response = await agent.invoke_async(user_input)

                # 7. Extract text content from response
                if hasattr(response, "content") and response.content:
                    if isinstance(response.content, str):
                        return response.content
                    elif isinstance(response.content, list):
                        text_blocks = [
                            block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                            for block in response.content
                        ]
                        return "\n".join(text_blocks).strip()

                return str(response)

    except Exception as e:
        logger.error(f"Error handling request in invoke(): {e}", exc_info=True)
        return f"An error occurred while processing your request: {str(e)}"


# ── CLI Entrypoint ────────────────────────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
