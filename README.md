# Retrieval-Augmented Generation (RAG) Pipeline with LangChain, Pinecone, and OpenAI Embeddings

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using [LangChain](https://python.langchain.com/), [Pinecone](https://www.pinecone.io/), and [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings). It processes a PDF document, indexes it using vector embeddings, and enables question-answering with a Large Language Model (LLM) and retrieved context.

## Features

- **PDF Ingestion**: Loads and parses PDF files.
- **Text Chunking**: Splits documents into overlapping chunks for efficient retrieval.
- **Embedding Generation**: Uses OpenAI's `text-embedding-3-large` model to embed text chunks.
- **Vector Database**: Stores embeddings in Pinecone for fast similarity search.
- **Contextual Retrieval**: Retrieves relevant chunks for a user query.
- **LLM Integration**: Uses Google's Gemini model to generate answers conditioned on retrieved context.
- **Prompt Engineering**: Supports both custom and LangChain Hub prompts.

## Pipeline Steps

1. **Load PDF Document**: Extracts text from a PDF using `PyPDFLoader`.
2. **Split Text into Chunks**: Uses `RecursiveCharacterTextSplitter` to create overlapping text segments.
3. **Create Embeddings**: Embeds text chunks with OpenAIEmbeddings.
4. **Set Up Pinecone**: Connects to Pinecone, checks/creates index, and loads it.
5. **Store Embeddings**: Adds embedded chunks to Pinecone Vector Store.
6. **Query Vector Store**: Finds relevant document chunks via similarity search.
7. **Create Retriever**: Sets up a retriever for fetching relevant context.
8. **Prompt Preparation**: Loads or defines a prompt template for RAG.
9. **LLM Setup**: Loads the Gemini model for answer generation.
10. **RAG Pipeline**: Chains together retrieval, prompting, LLM, and output parsing for end-to-end Q&A.

## Getting Started

### Prerequisites

- Python 3.8+
- API keys for [OpenAI](https://platform.openai.com/) and [Pinecone](https://www.pinecone.io/)
- Access to [Google Gemini](https://ai.google.dev/) (via LangChain)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/UtkPatAI25/RAG_LangChain_Pinecone_OpenAI_Embedding.git
   cd RAG_LangChain_Pinecone_OpenAI_Embedding
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - `OPENAI_API_KEY` (for OpenAI embeddings)
   - `PINECONE_API_KEY` (for Pinecone vector database)
   - `GOOGLE_API_KEY` (for Gemini LLM, if required)

   You can set them in your shell, or create a `.env` file.

   Example:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   export PINECONE_API_KEY=your_pinecone_api_key
   export GOOGLE_API_KEY=your_gemini_api_key
   ```

4. **Add your PDF:**
   - Place your PDF (e.g., `GenAI_Report_2023_011124.pdf`) in the `Data/` directory.

### Usage

Run the notebook:

```bash
jupyter notebook app.ipynb
```

Follow the cells and modify the queries as needed.

### Example Queries

- "What are the key findings of the report?"
- "What is the analysis all about?"

## File Structure

- `app.ipynb`: Main project notebook containing the complete code and pipeline.
- `requirements.txt`: Python dependencies for the project.
- `Data/`: Directory to store your PDF files.
- `README.md`: Project documentation.

## Notes

- Ensure your API keys are valid and have sufficient quota.
- The current configuration uses the `gemini-1.5-flash` model (Google Gemini) and OpenAI's large embedding model.
- Pinecone index is automatically created if it does not exist.

## References

- [LangChain Documentation](https://python.langchain.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Google Gemini](https://ai.google.dev/)

---

**Author:** [UtkPatAI25](https://github.com/UtkPatAI25)
