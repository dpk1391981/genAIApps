import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import pytube

#Stream lit app
st.title("Summarize URL content from YT or Website")
st.subheader("Summarize URL")


#Get Groq Key
with st.sidebar:
    groq_key = st.text_input("Groq API key", value="", type="password")
    
ge_url = st.text_input("URL:", label_visibility="collapsed")


if st.button("Summarize the content from YT or Website"):
    #validate
    if not groq_key.strip() or not ge_url.strip():
        st.error("Please provide info!")
        
    elif not validators.url(ge_url):
        st.error("Please provide valide URL.")
        
    else:
        try:
            #llm model
            llm=ChatGroq(groq_api_key=groq_key, model="gemma2-9b-it")

            ## prompt
            prompt_template = """
            Please provide a summary of the following content in 300 words:
            Summary: {text}
            """

            prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
            with st.spinner("Waiting ..."):
                if "youtube.com" in ge_url:
                    loader=YoutubeLoader.from_youtube_url(youtube_url=ge_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[ge_url], ssl_verify=False, headers={})
                docs = loader.load()
                
                #chain summerization
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)
                st.success(output_summary)
        except Exception as ex:
            st.error(f"Something went wrong! {ex}") 
            