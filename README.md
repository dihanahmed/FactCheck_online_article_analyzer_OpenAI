# 📰 FactCheck – News Research Tool

FactCheck is an AI-powered application that helps users extract reliable, source-grounded answers from any news article or publicly available web page. Built with LangChain, OpenAI APIs, FAISS, and Streamlit, this project demonstrates a complete Retrieval-Augmented Generation (RAG) pipeline for document-based question answering.

---

## 🚀 Features

* Accepts 1–2 URL inputs for article scraping
* Extracts and chunks article text using LangChain utilities
* Converts chunks into embeddings using OpenAI's `text-embedding-ada-002`
* Stores embeddings in a local FAISS vector database
* Uses OpenAI's `gpt-3.5-turbo` to answer user queries based on retrieved chunks
* Displays both the answer and its source URLs
* Streamlit interface for ease of use

---

## 🛠️ Tech Stack

* **Python 3.11**
* **LangChain**
* **OpenAI API** (`text-embedding-ada-002`, `gpt-3.5-turbo`)
* **FAISS** (Meta’s vector store for similarity search)
* **Streamlit** (frontend UI)
* **dotenv** (for secure API key storage)

---

## 📁 Project Structure

```
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── faiss_store_openai/     # Saved FAISS index files
├── .env                    # OpenAI API key (not committed)
```

---

## ⚙️ Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/factcheck.git
cd factcheck
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Add OpenAI API Key**
   Create a `.env` file:

```
OPENAI_API_KEY=your-openai-api-key-here
```

4. **Run the App**

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. User enters article URLs in the sidebar.
2. Content is scraped using `UnstructuredURLLoader`.
3. The text is split using `RecursiveCharacterTextSplitter`.
4. Chunks are embedded via OpenAI and stored in FAISS.
5. When the user asks a question, similar chunks are retrieved.
6. A prompt is constructed and passed to the LLM.
7. The model returns an answer along with its sources.

---

## 🧪 Example Use Case

* URL 1: [https://en.wikipedia.org/wiki/Sagittarius_A*](https://en.wikipedia.org/wiki/Sagittarius_A*)
* URL 2: [https://www.news.com.au/finance/real-estate/sydney-nsw/home-prices-in-2030](https://www.news.com.au/finance/real-estate/sydney-nsw/home-prices-in-2030-what-houses-or-units-will-cost-in-each-suburb/news-story/f95522d65de0630349cce2a804862f7d)
* Question: "What are the key findings in the second article?"

---

## 🔒 Security Notes

* The `.env` file stores sensitive API keys. Never commit this file.
* All user inputs are processed in-memory; no data is saved or logged.

---

## 📌 Limitations

* Does not support paywalled or JavaScript-heavy websites.
* Depends on OpenAI API limits and pricing.
* Works best with informative articles (news, blogs, reports).

---

## 🌱 Future Improvements

* Add PDF and DOCX support
* Switch to open-source LLMs (e.g., Mistral, LLaMA)
* Enable summarization and chat memory

---

## 📄 License

This project is for educational and research purposes.

---

## 🙌 Acknowledgments

* [LangChain](https://www.langchain.com)
* [OpenAI](https://platform.openai.com)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Streamlit](https://streamlit.io)
