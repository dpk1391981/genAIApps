from crewai_tools import YoutubeChannelSearchTool
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from crewai_tools import tool
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI

api_key = "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"
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

mysql_host = "localhost"
mysql_user =  "genai"
mysql_password = "genai"
mysql_db = "offer_prm_uat"

# Initialize the tool with a specific Youtube channel handle to target your search
yt_tool = YoutubeChannelSearchTool(youtube_channel_handle='@DeepakKumar-qw7hg')

def config_mysql_db(mysql_host, mysql_user, mysql_password, mysql_db):
    """Configure MySQL Database connection."""
    if not (mysql_db and mysql_host and mysql_user and mysql_password):
        return False
    
    db_engine = create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}")
    db = SQLDatabase(db_engine)
    
    # Restrict tables to a specific subset
    # db._inspector.get_table_names = lambda: ["facility" ]
    return db

db = config_mysql_db(mysql_host, mysql_user, mysql_password, mysql_db)

@tool("tables_schema")
def tables_schema(tables: str) -> str:
    """
    Input is a comma-separated list of tables, output is the schema and sample rows
    for those tables. Be sure that the tables actually exist by calling `list_tables` first!
    Example Input: table1, table2, table3
    """
    
    
    tool = InfoSQLDatabaseTool(db=db)
    return tool.invoke(tables)

@tool("list_tables")
def list_tables() -> str:
    """List the available tables in the database"""
    return ListSQLDatabaseTool(db=db).invoke("")


@tool("execute_sql")
def execute_sql(sql_query: str) -> str:
    """Execute a SQL query against the database. Returns the result."""
    return QuerySQLDataBaseTool(db=db).invoke(sql_query)


@tool("check_sql")
def check_sql(sql_query: str) -> str:
    """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with `execute_sql`.
    """
    return QuerySQLCheckerTool(db=db, llm=llm).invoke({"query": sql_query})