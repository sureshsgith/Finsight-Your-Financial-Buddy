# Author: Suresh Ratlavath
# Email: srdev3175@gmail.com
# Date: 21-08-2023

# Import statements
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
import os
import streamlit as st
import time
import json
from pathlib import Path
from pprint import pprint
from langchain.retrievers.web_research import WebResearchRetriever
import asyncio
from decouple import config
# Load environment variables from the .env file
OPENAI_API_KEY = config('OPENAI_API_KEY')
GOOGLE_CSE_ID = config('GOOGLE_CSE_ID')
GOOGLE_API_KEY = config('GOOGLE_API_KEY')
ELEVENLABS_API_KEY = config('ELEVENLABS_API_KEY')

# Set the environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GOOGLE_CSE_ID"] = GOOGLE_CSE_ID
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Streamlit app setup
st.markdown("<h1 align=center>💲FinSight - Your Finance Buddy📈</h1>", unsafe_allow_html=True)

# Importing necessary libraries
import langchain
from elevenlabs import set_api_key
set_api_key(ELEVENLABS_API_KEY)
from elevenlabs import generate, play
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chains import RetrievalQAWithSourcesChain

# Function for web scraping tool
def web_scraping_tool(query: str) -> str:
    # Vectorstore
    vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai")

    # LLM
    llm = ChatOpenAI(temperature=0)

    # Search
    search = GoogleSearchAPIWrapper()

    # Initialize
    web_research_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore,
        llm=llm,
        search=search,
    )
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_research_retriever)
    result = qa_chain({"question": query})
    return result



# List of available tools
tools = [
    Tool(
        name="WebScraping",
        func=web_scraping_tool,
        description="Useful for when you need to answer questions about current events and market data. Use this tool to get current information about Financial investment and more. and also Useful for scraping information of Financial and market from websites and internet"
    ),
]

# Defining chat prompts
# (Please note that prompt variables are repeated and will need to be maintained in sync)
prompt = """Hey AI your name is Sunny,Your Financial Buddy, please act as if you're my close friend, not a professional, and let's talk about my financial goals and plans. Your tone should be warm, friendly, and reassuring, just like a trusted friend. Feel free to guide me through financial decisions and offer advice as you would to a close buddy. You can also access this tool to get more better responses.

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be {tool_names}
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

These were previous tasks you completed:
{history}
Begin!

Question: {input}
{agent_scratchpad}"""

if "messages" not in st.session_state:
        st.session_state.messages = []

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["history"] = st.session_state.messages
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

# Creating a custom prompt
prompt = CustomPromptTemplate(
    template=prompt,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)

# Custom output parser
class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if match:
            action = match.group(1).strip()
            action_input = match.group(2).strip(" ").strip('"')
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)

        if "\nObservation:" in llm_output:
            observation = llm_output.split("\nObservation:", 1)[-1].strip()
            return AgentAction(tool="Observation", tool_input="", log=llm_output)

        return AgentFinish(
            return_values={"output": llm_output.strip()},
            log=llm_output,
        )

# Creating an output parser
output_parser = CustomOutputParser()

# Initializing ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.7)

# Creating an LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# List of tool names
tool_names = [tool.name for tool in tools]

# Creating an LLMSingleActionAgent
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

# Creating an AgentExecutor
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# Streamlit chat interface

for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
if prompt := st.chat_input("Ask a Question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        ai_response = agent_executor.run(prompt)
        audio = generate(
            text=ai_response,
            voice="Bella",
            model="eleven_monolingual_v1"
        )
        play(audio)
        message_placeholder.write(ai_response)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
