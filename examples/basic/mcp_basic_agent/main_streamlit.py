import asyncio
import os
import time
import logging
import streamlit as st
import requests

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

# url = "https://firefall-stage.adobe.io/v1/chat/completions"
# IMS_URL = "https://firefall-stage.adobe.io"
# IMS_AUTH_CODE = 'eyJhbGciOiJSUzI1NiIsIng1dSI6Imltc19uYTEtc3RnMS1rZXktcGFjLTEuY2VyIiwia2lkIjoiaW1zX25hMS1zdGcxLWtleS1wYWMtMSIsIml0dCI6InBhYyJ9.eyJpZCI6ImNvZGVBSV9zdGciLCJ0eXBlIjoiYXV0aG9yaXphdGlvbl9jb2RlIiwiY2xpZW50X2lkIjoiY29kZUFJIiwidXNlcl9pZCI6ImNvZGVBSUBBZG9iZVNlcnZpY2UiLCJhcyI6Imltcy1uYTEtc3RnMSIsIm90byI6ZmFsc2UsImNyZWF0ZWRfYXQiOiIxNjk2ODg2Njk3NjE5Iiwic2NvcGUiOiJzeXN0ZW0ifQ.0FTsZIP_Pm6nRUDTSs_Ntuc0lZostWfiNUvyCK9B6J2G5fAqDH3yCzfdEJB2JYz_PmaJ-l4nBxAfT_4Ul-g_bXbjEEEEuHjOHEIOp8WtA-vx636C8lFdomr_PNkbimVDmRDj-CJ2bMKVcWm7evI7HKY511Xja5zDEgDnCFNoH6E7Lq3ReRd3eP-jo-SpfZmNCk9tRu16SiVfqIjh6RLZpOEcYMnIXbAG4DwOF-I1cZl5qagWiolHhJu0KkyA_75oC-9KUduiy8eI2pauegqrEQJX_7rTqOdkEJgySRngscfGRZqd9w2ouJtshUTs7RxNd4zsmHm8PxL30I5KaspYWQ'
# IMS_CLIENT_ID = 'codeAI'
# IMS_CLIENT_SECRET = 's8e-JpCE64IBBBeJL6st2cQ7ENb7XkNXxVMH'
# IMS_ORG_ID = "154340995B76EEF60A494007@AdobeOrg"
# model_name = "gpt-4o-mini"
# llm_type = "azure_chat_openai"
# temperature = 0.0
# tokens = 4096

# def generate_token(client_id, client_secret, code):
#     headers = {
#         'Accept': '*/*',
#         'Content-Type': 'application/x-www-form-urlencoded'
#     }
#     authorization_code = 'authorization_code'
#     response = requests.post(f'https://ims-na1-stg1.adobelogin.com/ims/token/v1?grant_type=authorization_code&client_id={client_id}&client_secret={client_secret}&&code={code}', headers=headers)
#     token_data = response.json()
#     print(token_data)
#     return token_data['access_token']

# token = generate_token(IMS_CLIENT_ID, IMS_CLIENT_SECRET, IMS_AUTH_CODE)


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
            "adobe-wiki": MCPServerSettings(
                command="node",
                args=["/Users/gaohuaz/Desktop/Work_station/GitHub/adobe-mcp-servers/src/adobe-wiki/dist/index.js"],
            ),
        }
    ),
    openai=OpenAISettings(
        api_key= "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",
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
            server_names=["fetch", "filesystem", "adobe-wiki"],
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
                message="Summarize the content you just retrieved in a 128 character tweet, and then use the adobe-wiki tool to summarized the content of https://wiki.corp.adobe.com/x/YOUB0g",
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
