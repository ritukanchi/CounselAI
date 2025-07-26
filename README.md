# RAG Legal Chatbot

This project implements an intelligent AI chatbot designed to provide accurate and reasoned legal information, primarily focusing on the Bharatiya Nyaya Sanhita (BNS) and other Indian legal frameworks. It leverages a Retrieval-Augmented Generation (RAG) architecture with advanced agentic capabilities and semantic caching for efficient and reliable responses.

## Features

* **Legal Information Retrieval:** Answers questions based on indexed legal documents (BNS, IPC, etc.).
* **Reasoned Responses:** Provides structured answers including relevant legal sections, detailed punishments with context, and practical legal advice.
* **Conversational Memory:** Maintains context across multiple turns for natural dialogue.
* **Intelligent Tool Use:** Dynamically decides to use its internal legal knowledge base, perform real-time web searches, or consult Wikipedia based on the query.
* **Performance Optimization:** Utilizes semantic caching to speed up responses for similar queries.
* **Quality Assurance:** Integrates an evaluation framework (Ragas) to measure and improve response accuracy and relevance.

## Technologies Used

* **Programming Language:** Python
* **LLMs:** Google Gemini (Embeddings), Groq (for Llama 3 & Mixtral inference)
* **Orchestration Framework:** Langchain (Agents, Chains, Tools, Prompts)
* **Vector Database:** Qdrant
* **Frontend:** Streamlit
* **Web Scraping:** requests, aiohttp, BeautifulSoup
* **Evaluation:** ragas
* **Secret Management:** python-dotenv
* **External Tools:** Tavily Search, Wikipedia

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd RAG_legal_chatbot
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    Create a `.env` file in the root directory of your project and add your API keys and service URLs:
    ```
    GEMINI_API_KEY=
    GROQ_API_KEY=
    QDRANT_CLOUD_URL=
    QDRANT_CLOUD_API_KEY=
    TAVILY_API_KEY=
    ```



## How to Run

Once the setup is complete and data is ingested, run the Streamlit application from the project root:

```bash
streamlit run app.py