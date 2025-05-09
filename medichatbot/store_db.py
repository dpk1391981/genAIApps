from src.helper import load_pdf_file, text_split, download_huggingface_embedding
from langchain.vectorstores.cassandra import Cassandra
import cassio
from dotenv import load_dotenv
import os
load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID_MULTI_AGENT = os.getenv("ASTRA_DB_ID_MULTI_AGENT")


cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID_MULTI_AGENT)

#extract data from pdf
extract_data = load_pdf_file("Data/")
#split text
split_data = text_split(extract_data)
#embeddings
embeddings = download_huggingface_embedding()



#store data in
def store_data(data, embeddings):
    vector_store = Cassandra(
        embedding=embeddings,
        table_name="medi_tbl",
        session=None,
        keyspace=None
    )
    vector_store.add_documents(documents=data)
    return "Data stored successfully"

store = store_data(extract_data, embeddings)
print(store)