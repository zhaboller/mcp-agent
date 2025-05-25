import asyncio
import os
import time
import logging

from mcp_agent.app import MCPApp
from mcp_agent.config import (
    Settings,
    LoggerSettings,
    MCPSettings,
    MCPServerSettings,
    OpenAISettings,
    AnthropicSettings,
)
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences

# from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

settings = Settings(
    execution_engine="asyncio",
    logger=LoggerSettings(type="file", level="info"),
    mcp=MCPSettings(
        servers={
            "fetch": MCPServerSettings(
                command="uvx",
                args=["mcp-server-fetch"],
            ),
            "filesystem": MCPServerSettings(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem"],
            ),
        }
    ),
    openai=OpenAISettings(
        api_key="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",
        default_model="gpt-4o-mini",
    ),
    anthropic=AnthropicSettings(
        api_key="sk-my-anthropic-api-key",
    ),
)

# Settings can either be specified programmatically,
# or loaded from mcp_agent.config.yaml/mcp_agent.secrets.yaml
app = MCPApp(name="mcp_basic_agent", settings=settings)  # settings=settings)


def get_user_input():
    """Get user input for file or URL to process"""
    print("\n" + "=" * 50)
    print("Interactive MCP Agent")
    print("=" * 50)
    print("You can provide either:")
    print("1. A file path (e.g., 'config.yaml', '/path/to/file.txt')")
    print("2. A URL (e.g., 'https://example.com/page')")
    print("=" * 50)

    user_input = input("Enter file path or URL: ").strip()

    if not user_input:
        print("No input provided. Exiting...")
        return None

    return user_input


def determine_input_type(user_input):
    """Determine if input is a URL or file path"""
    if user_input.startswith(("http://", "https://")):
        return "url"
    else:
        return "file"


async def example_usage():
    # Get user input first
    user_input = get_user_input()
    if not user_input:
        return

    input_type = determine_input_type(user_input)
    print(f"\nDetected input type: {input_type}")
    print(f"Processing: {user_input}")
    print("-" * 50)

    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        logger.info("Current config:", data=context.config.model_dump())

        # Add the current directory to the filesystem server's args
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        finder_agent = Agent(
            name="finder",
            instruction="""You are an agent with access to the filesystem, 
            as well as the ability to fetch URLs. Your job is to identify 
            the closest match to a user's request, make the appropriate tool calls, 
            and return the URI and CONTENTS of the closest match.""",
            server_names=["fetch", "filesystem"],
        )

        async with finder_agent:
            logger.info("finder: Connected to server, calling list_tools...")
            result = await finder_agent.list_tools()
            logger.info("Tools available:", data=result.model_dump())

            llm = await finder_agent.attach_llm(OpenAIAugmentedLLM)

            # Process based on input type
            if input_type == "file":
                # Handle file input
                message = f"Print the contents of {user_input} verbatim"
                print(f"Requesting file contents: {user_input}")
            else:
                # Handle URL input
                message = f"Print the first 2 paragraphs of {user_input}"
                print(f"Fetching URL content: {user_input}")

            result = await llm.generate_str(message=message)

            if input_type == "file":
                logger.info(f"File contents ({user_input}): {result}")
                print(f"\nFile Contents:\n{'-'*30}\n{result}")
            else:
                logger.info(f"First 2 paragraphs of {user_input}: {result}")
                print(f"\nURL Content (first 2 paragraphs):\n{'-'*30}\n{result}")

            # Multi-turn conversation: summarize the content
            print(f"\n{'='*50}")
            print("Generating summary...")
            print("=" * 50)

            result = await llm.generate_str(
                message="Summarize the content you just retrieved in a 128 character tweet",
                request_params=RequestParams(
                    modelPreferences=ModelPreferences(
                        costPriority=0.1, speedPriority=0.2, intelligencePriority=0.7
                    ),
                ),
            )
            logger.info(f"Content as a tweet: {result}")
            print(f"Summary Tweet:\n{'-'*15}\n{result}")


if __name__ == "__main__":
    start = time.time()
    asyncio.run(example_usage())
    end = time.time()
    t = end - start

    print(f"\nTotal run time: {t:.2f}s")
