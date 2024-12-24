import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()

import chromadb;
chromadb.api.client.SharedSystemClient.clear_system_cache()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
groq_api_key=os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# setup streamlit
st.title("Conversational RAG with PDF upload and chat history")
st.write("Upload PDFs and chat with their content")

# input key for GROQ
api_key=st.text_input("Enter your Groq API key: ", type="password")
if api_key:
    llm=ChatGroq(groq_api_key=api_key, model="gemma2-9b-it")
    
    ##chat interface
    session_id=st.text_input("Session ID", value="default_session")
    
    if 'store' not in st.session_state:
        st.session_state.store={}
        
    upload_files=st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    print(f"uploaded file: {upload_files}")
    if upload_files:
        documents=[]
        for uploaded_file in upload_files:
            tempdf=f"./tmp.pdf"
            with open(tempdf, "wb") as file:
                file.write(uploaded_file.getvalue())  
                file_name=uploaded_file.name
                
            loader=PyPDFLoader(tempdf)
            docs=loader.load()
            documents.extend(docs)
            
        #splitting the text and storing in vector db
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        splits=text_splitter.split_documents(documents=documents)
        vectors=Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever=vectors.as_retriever()
        
        #history retriever prompt
        contextual_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might refrence context in the chat history"
            "formulate a standlone quesiton which can be understood"
            "without the chat history, Do not the answer of question"
            "just reformulate it if needed and otherwise return it as it"
        )
        
        contextual_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system", contextual_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        history_aware_retriver=create_history_aware_retriever(llm, retriever,contextual_q_prompt)
        
        #Answer
        sytem_prompt=(
            "You are an assistent for question and answer the task"
            "use the following piece of retrieved context to answer"
            "the question , you don't now the answer, say that you"
            "don't know . Use three sentences maximum and keep the "
            "answer concise"
            "\n\n"
            "{context}"
        )
        
        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system", sytem_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriver,question_answer_chain)
        
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        user_input=st.text_input("Enter you query:")
        if user_input:
            sessio_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                }, 
            )
            
            # st.write(st.session_state.store)
            st.write("**Assistence:** ",response["answer"])
            # st.write("Chat history: ", sessio_history.messages)
else: 
    st.warning("Please enter you Groq Id!")