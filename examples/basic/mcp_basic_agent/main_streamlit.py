import asyncio
import os
import time
import logging
import streamlit as st

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
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

# Initialize session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = None

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
        api_key="sk-my-openai-api-key",
        default_model="gpt-4o-mini",
    ),
    anthropic=AnthropicSettings(
        api_key="sk-my-anthropic-api-key",
    ),
)

app = MCPApp(name="mcp_basic_agent", settings=settings)

def determine_input_type(user_input):
    """Determine if input is a URL or file path"""
    if user_input.startswith(("http://", "https://")):
        return "url"
    else:
        return "file"

async def process_input(user_input):
    """Process the user input and return results"""
    if not user_input:
        return None

    input_type = determine_input_type(user_input)
    
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context
        logger.info("Current config:", data=context.config.model_dump())
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

            if input_type == "file":
                message = f"Print the contents of {user_input} verbatim"
            else:
                message = f"Print the first 2 paragraphs of {user_input}"

            content_result = await llm.generate_str(message=message)
            
            # Generate summary
            summary_result = await llm.generate_str(
                message="Summarize the content you just retrieved in a 128 character tweet",
                request_params=RequestParams(
                    modelPreferences=ModelPreferences(
                        costPriority=0.1, speedPriority=0.2, intelligencePriority=0.7
                    ),
                ),
            )
            
            return {
                'input_type': input_type,
                'content': content_result,
                'summary': summary_result
            }

def main():
    st.title("MCP Agent Interface")
    st.write("Enter either a file path or URL to process:")
    
    user_input = st.text_input("File path or URL:")
    
    if st.button("Process"):
        if user_input:
            with st.spinner("Processing..."):
                results = asyncio.run(process_input(user_input))
                st.session_state.results = results
        else:
            st.warning("Please enter a file path or URL")
    
    if st.session_state.results:
        st.subheader("Results")
        
        if st.session_state.results['input_type'] == 'file':
            st.write("File Contents:")
        else:
            st.write("URL Content (first 2 paragraphs):")
            
        st.text_area("Content", st.session_state.results['content'], height=200)
        
        st.subheader("Summary Tweet")
        st.write(st.session_state.results['summary'])

if __name__ == "__main__":
    main()
