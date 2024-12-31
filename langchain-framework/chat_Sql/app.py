import streamlit as st
from langchain.agents import initialize_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain.prompts import SemanticSimilarityExampleSelector,PromptTemplate,FewShotPromptTemplate
from few_shots import few_shots
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()

# Streamlit setup
st.title("Clove PRM AI Assistent")

# Sidebar inputs for MySQL connection and Groq API Key
mysql_host = st.sidebar.text_input("MySQL Host", "localhost")
mysql_user = st.sidebar.text_input("MySQL User", "root")
mysql_password = st.sidebar.text_input("MySQL Password", type="password")
mysql_db = st.sidebar.text_input("MySQL DB", "")
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# Query limit setting
query_limit = st.sidebar.number_input(
    "Set maximum number of rows to fetch per query:",
    min_value=1,
    max_value=1000,
    value=100,
    step=1,
)

# Validate required fields
if not mysql_db:
    st.info("Enter MySQL Database name.")
if not api_key:
    st.info("Enter Groq API Key.")
    
llm=None
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model="gemma2-9b-it", streaming=True)

@st.cache_resource(ttl="2h")
def config_mysql_db(mysql_host, mysql_user, mysql_password, mysql_db):
    """Configure MySQL Database connection."""
    if not (mysql_db and mysql_host and mysql_user and mysql_password):
        st.error("Please provide complete MySQL DB configuration.")
        st.stop()
    return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

# Configure the database
db = config_mysql_db(mysql_host, mysql_user, mysql_password, mysql_db)

# Toolkit setup
toolkit = SQLDatabaseToolkit(llm=llm, db=db)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
to_vectorize = [" ".join(example.values()) for example in few_shots]
vectorstore = FAISS.from_texts(to_vectorize, embeddings, metadatas=few_shots)
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)


mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".
    
    Use the following format:
    
    Question: Question here
    SQLQuery: Query to run with no pre-amble
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    
    No pre-amble.
    """

example_prompt = PromptTemplate(
    input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
    template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
)

few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
    )

# Initialize the agent with few-shot prompting
agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,  # Enable parsing error handling
    prompt=few_shot_prompt,  # Use the FewShotPromptTemplate
)

# Chat history initialization
if "messages" not in st.session_state or st.sidebar.button("Clear History"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
user_input = st.chat_input("Ask from your local database!")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        try:
            # Add LIMIT clause to user input
            limited_query = f"{user_input} LIMIT {query_limit}"
            streamlit_callback = StreamlitCallbackHandler(st.container())
            response = agent.run(limited_query, callbacks=[streamlit_callback])
            response = response[:500]


            # Save assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
        except ValueError as e:
            st.error(f"Parsing Error: {e}")
        except Exception as e:
            st.error(f"Error: {e}")
