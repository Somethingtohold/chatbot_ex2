# Data Science & Technology Career Advisor Bot

An AI career advisor built for **Turing College Sprint 2** project. The bot provides career guidance in data science and technology. It uses RAG and tool calling. 

## Overview

The bot helps users explore data science and technology career paths by combining:
- A curated knowledge base (roles, career paths, salary benchmarks, WEF Future of Jobs 2025 report)
- Live data via external APIs
- Conversational AI by GPT-4o-mini

## Technical Implementation

### Advaned RAG with Query Translation
- Knowledge base stored in **ChromaDB** vector database
- Documents embedded using **OpenAI text-embedding-3-small**
- **MultiQueryRetriever** generates multiple query variants to improve retrieval recall
- Auto-rebuilds vector store when knowledge base files change

### Tool Calling
| Tool | API | Description |
|------|-----|-------------|
| Salary Estimator | Adzuna Histogram API | Average salary for a role in a European country |
| Recent Job Openings | Adzuna Search API | 5 most recent job listings for a role and location |
| Course Recommender | YouTube Data API v3 | Top tutorial videos for a given skill. Tried Coursera and Udemy APIs but they were not available / maintained anymore / got blocked. |

### Other Features
- **LangChain** agent with `create_openai_tools_agent`
- **Token monitoring** with tiktoken (`o200k_base` encoding)
- **Rate limiting**: 20 requests per session
- **Input validation** and error handling

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/chatbot_ex2.git
cd chatbot_ex2
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure environment variables**
```bash
cp .env.example .env
```
Fill in your API keys in `.env`.

**4. Ingest the WEF report** (first run only)
```bash
python ingest_wef.py
```

**5. Run the app**
```bash
streamlit run app.py
```

## Required API Keys

- `OPENAI_API_KEY` — [platform.openai.com](https://platform.openai.com)
- `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` — [developer.adzuna.com](https://developer.adzuna.com)
- `YOUTUBE_API_KEY` — [console.cloud.google.com](https://console.cloud.google.com)


## Applied improvements / learnings from previous Sprint feedback


| No. | Previous mistake | Improvement |
|-----:|------------------|-------------|
| 1 | Missing .gitignore (not committed) | Addded a project .gitignore with sensible entries and committed it to the repo. |
| 2 | Dependency versions not pinned | Pinned dependencies to known-good exact versions in requirements.txt. |
| 3 | Approximate token estimation | Use tiktoken to compute accurate token counts for prompts and responses. |
| 4 | No privacy or data‑retention notice | Added a brief privacy/data‑retention notice in the UI describing what data is stored, retention period, and user opt-out/contact info. |
| 5 | Opaque system prompt set-up | Wrapped in try/except. |
| 6 | Single-file application limits maintainability | Refactored into a small package (separate app, tools, RAG, ingestion modules) as shown in the Project Structure. |


## Limitations
- I did not deploy to cloud with proper scaling.
- I did not optimise my RAG set-up, e.g. not performed RAGAs.
- My knowledge bases are static / they do not get updated automatically.
- My tool supports only one model - OpenAI. No multi-model.


## Project Structure

```
chatbot_ex2/
├── app.py                  # Streamlit application
├── tools.py                # LangChain tools (salary, jobs, courses)
├── rag.py                  # RAG pipeline with MultiQueryRetriever
├── ingest_wef.py           # WEF Future of Jobs 2025 report ingestion
├── system_prompt.txt       # System prompt
├── knowledge_base/         # Domain knowledge files
│   ├── career_paths.txt
│   ├── data_science_roles.txt
│   ├── salary_data.txt
│   └── wef_future_of_jobs_2025.txt
├── requirements.txt
└── .env.example
```
