from src.helper import load_pdf_file, text_split, download_huggingface_embedding
from langchain.vectorstores.cassandra import Cassandra
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from dotenv import load_dotenv
import gradio as gr
import os
import cassio
 
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID_MULTI_AGENT = os.getenv("ASTRA_DB_ID_MULTI_AGENT")

embeddings = download_huggingface_embedding()
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID_MULTI_AGENT)

vectorstore = Cassandra(
    embedding=embeddings,
    session=None,
    keyspace="default_keyspace",
    table_name="medi_tbl"
)

retriever = vectorstore.as_retriever()


llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0.5, max_completion_tokens=500)

prompt  = ChatPromptTemplate.from_messages(
    [
        ("system",sytem_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def chat(message, history=None):
    response = rag_chain.invoke({"input":message})
    return str(response["answer"])

with gr.Blocks() as app:
    gr.Markdown("# Medical AI Assistant")

    gr.ChatInterface(chat)

if __name__ == "__main__":
    app .launch()