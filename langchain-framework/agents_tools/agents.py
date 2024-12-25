import streamlit as st
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# import os
# from dotenv import load_dotenv
# load_dotenv()


api_wrapper_wiki= WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv= ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search=DuckDuckGoSearchRun(name="Search")

st.title("Chat with search")

## side bar setting
st.sidebar.title("Settins")
api_key = st.sidebar.text_input("Groq API Key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
if api_key: 
    if prompt:=st.chat_input(placeholder="Enter you query!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")
        tools = [wiki, arxiv, search]

        search_agent=initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
        
        with st.chat_message("assistant"):
            st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            respose=search_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": respose})
            st.write(respose)
else:
    st.warning("Please enter Groq API Key")