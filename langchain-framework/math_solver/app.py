import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv
load_dotenv()
Hf_token=os.getenv("HF_TOKEN")

st.set_page_config(page_title="Text to math solver")
st.title("Text to math solver")
print(Hf_token)
groq_key = st.sidebar.text_input("Groq API key", value=Hf_token, type="password")
if not groq_key:
    st.error("Please provide Groq API key!")
    st.stop()

# Initialize the model
# llm=ChatGroq(groq_api_key=groq_key, model="gemma2-9b-it")

#llm by hugging face
repo_id="mistralai/Mistral-7B-Instruct-v0.3"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=150, temperature=0.7, token=Hf_token)

#initialize the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(name="Wikipedia", func=wikipedia_wrapper.run, description="Search Wikipedia for information")

#initialize the agent for math tool
math_chain=LLMMathChain.from_llm(llm=llm)
calculater = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Calculate the math expression"
)

#prompt
prompt = """
You are agent tasked for solving users mathmetical question. Logically, arrived at the solution and provide the detail explanation and display the point wise for the below question.
Question: {question}
Answer: 
"""
promp_tmp = PromptTemplate(input_variables=["question"], template=prompt)
    
#combine all tools
chain =LLMChain(llm=llm, prompt=promp_tmp)

#Reasoning tool
reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A tool for answering logic based and reasoning questions"
)

#init agent
assistent_agent = initialize_agent(
    tools=[wikipedia_tool, calculater, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state.messages=[{
        "role": "assistant",
        "content": "Hi, How can i help you!"
    }]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
#function to generate response

def generate_response(question):
    resposne  = assistent_agent.invoke({"input":question })
    return resposne

question=st.text_area("Enter you question:",  "")
if st.button("find my answer"):
    if question:
        with st.spinner("Generating response"):
            st.session_state.messages.append({"role": "user", "content":  question})
            st.chat_message("user").write(question)
            
            st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistent_agent.run(st.session_state.messages, callbacks=[st_cb])
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write("#Response")
            st.success(response)

    