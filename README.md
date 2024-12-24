# genAIApps

# Project Read the pdf file and Response based on given pdfs RAG Arch

## Environment Variables Configuration

To set up the project, you need to create a `.env` file in the root directory and define the following environment variables:

- **`LANGCHAIN_API_KEY`**: Your API key for LangChain. This is required for authenticating with the LangChain API.  
- **`LANGCHAIN_PROJECT`**: The name of the LangChain project you are working on.  
- **`LANGCHAIN_TRACING_V2`**: A boolean value (`true` or `false`) to enable or disable LangChain's advanced tracing features.  
- **`GROQ_API_KEY`**: Your API key for accessing the GROQ service.  
- **`HF_TOKEN`**: The Hugging Face token for accessing Hugging Face's APIs or resources.

### Example `.env` File

```plaintext
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_PROJECT=your_project_name
LANGCHAIN_TRACING_V2=true
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_hf_token
```
