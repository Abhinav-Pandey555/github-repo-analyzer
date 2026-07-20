"""
GitHub Repo Analyzer - Streamlit app
Requires: streamlit, requests, python-dotenv, pandas, PyGithub, faiss-cpu,
          langchain, langchain-core, langchain-community, langchain-google-genai

Env vars required (.env or system):
    GEMINI_API_KEY
    GITHUB_TOKEN
"""

import os
import time
import random
import streamlit as st
from dotenv import load_dotenv
from github import Github, GithubException
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

import utils.config as config
from utils.constants import *

# ── Load env variables ───────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Google has been deprecating/retiring Gemini models frequently in 2026
# (2.0 Flash retired March 2026, 2.5 Flash/Flash-Lite reportedly returning
# 404s ahead of their official Oct 2026 shutdown). Rather than hardcoding a
# model name that can break overnight, read it from .env with a current
# fallback, so a future deprecation is a one-line .env edit, not a code change.
# Check https://ai.google.dev/gemini-api/docs/models for the current list.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

# GoogleGenerativeAIEmbeddings / GoogleGenerativeAI are called WITHOUT an
# explicit google_api_key= param (see fix for the 504 Deadline Exceeded bug),
# so they read auth exclusively from the GOOGLE_API_KEY env var. Set it here.
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GitHub Repo Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Blue & Dark Theme ────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .stApp {
            background-color: #0d1117;
            color: #e6edf3;
        }
        [data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #1f6feb;
        }
        .stButton>button {
            background-color: #1f6feb;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 20px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #388bfd;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #161b22;
            color: #e6edf3;
            border: 1px solid #1f6feb;
            border-radius: 6px;
        }
        .stAlert {
            background-color: #161b22;
            border: 1px solid #1f6feb;
            border-radius: 6px;
        }
        h1, h2, h3 {
            color: #388bfd !important;
        }
        hr {
            border-color: #1f6feb;
        }
        [data-testid="stChatMessage"] {
            background-color: #161b22;
            border: 1px solid #1f6feb;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        [data-testid="stChatInput"] {
            background-color: #161b22;
            border: 1px solid #1f6feb;
            border-radius: 8px;
        }
        [data-testid="stMetric"] {
            background-color: #161b22;
            border: 1px solid #1f6feb;
            border-radius: 8px;
            padding: 12px;
        }
        [data-testid="stMetricLabel"] {
            color: #388bfd !important;
        }
        [data-testid="stMetricValue"] {
            color: #e6edf3 !important;
        }
    </style>
""", unsafe_allow_html=True)


# ── Token estimator ──────────────────────────────────────────────────────────
def estimate_tokens(text):
    return len(str(text)) // 4


# ── Retry helper with exponential backoff for rate-limit errors ─────────────
def call_with_retry(func, *args, max_retries=5, base_delay=3, **kwargs):
    """Retries a callable on rate-limit / transient errors with exponential backoff."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            is_rate_limited = any(k in msg for k in ["429", "resource_exhausted", "quota", "rate limit"])
            if is_rate_limited and attempt < max_retries - 1:
                wait = base_delay * (2 ** attempt) + random.uniform(0, 1)
                st.toast(f"⏳ Gemini API busy, retrying in {wait:.0f}s...", icon="⏳")
                time.sleep(wait)
                continue
            raise last_err
    raise last_err


# ── GitHub client (cached, single instance) ──────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_github_client():
    return Github(GITHUB_TOKEN) if GITHUB_TOKEN else Github()


# ── Fetch repos via PyGithub (single fetch, reused for display + analysis) ──
def fetch_user_repos(username, include_topics=False, max_repos=30):
    """Returns (repo_info_list, error_message). Exactly one will be None."""
    try:
        client = get_github_client()
        user = client.get_user(username)
        repos = list(user.get_repos())
    except GithubException as e:
        if e.status == 404:
            return None, "Invalid GitHub username. Please check and try again."
        if e.status == 403:
            return None, "GitHub API rate limit exceeded. Add a GITHUB_TOKEN to your .env or try again later."
        return None, f"GitHub API error: {e}"
    except Exception as e:
        return None, f"Unexpected error while fetching repos: {e}"

    repo_info = []
    for repo in repos:
        if repo.fork:
            continue
        topics = []
        if include_topics:
            try:
                topics = repo.get_topics()
            except Exception:
                topics = []
        repo_info.append({
            "name": repo.name,
            "description": repo.description,
            "language": repo.language,
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "open_issues": repo.open_issues_count,
            "topics": topics,
            "url": repo.html_url,
        })
        if len(repo_info) >= max_repos:
            break

    if not repo_info:
        return [], None

    return repo_info, None


# ── Display repos as clickable links ─────────────────────────────────────────
def display_repos(repo_info):
    for repo in repo_info:
        st.write(f"[{repo['name']}]({repo['url']})")


# ── Build FAISS vector store directly from repo dicts (no CSV round-trip) ───
def build_vector_store(repo_info):
    if not repo_info:
        return None, "No non-fork repositories found to analyze."

    docs = []
    for r in repo_info:
        content = (
            f"Repository: {r['name']}\n"
            f"Description: {r['description'] or 'No description'}\n"
            f"Language: {r['language'] or 'Unknown'}\n"
            f"Stars: {r['stars']}\n"
            f"Forks: {r['forks']}\n"
            f"Open Issues: {r['open_issues']}\n"
            f"Topics: {', '.join(r['topics']) if r['topics'] else 'None'}\n"
            f"URL: {r['url']}"
        )
        docs.append(Document(page_content=content, metadata={"name": r["name"], "url": r["url"]}))

    try:
        # NOTE: GOOGLE_API_KEY is already set as an env var at startup (from
        # GEMINI_API_KEY). Do NOT also pass google_api_key= here -- passing it
        # both ways has been reported to cause spurious 504 Deadline Exceeded
        # errors. transport="rest" avoids gRPC-related timeouts on
        # networks/firewalls/VPNs that interfere with gRPC, and request_options
        # raises the default 60s deadline.
        embeddings = GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL,
            transport="rest",
            request_options={"timeout": 120},
        )
    except Exception as e:
        return None, f"Failed to initialize embeddings: {e}"

    # ── Batch + throttle to respect free-tier RPM limits ────────────────────
    batch_size = 10
    vector_store = None
    try:
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]

            def _add_batch():
                nonlocal vector_store
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)

            call_with_retry(_add_batch)

            if i + batch_size < len(docs):
                time.sleep(4)  # stay comfortably under free-tier RPM
    except Exception as e:
        return None, f"Failed to build knowledge base (embedding error): {e}"

    return vector_store, None


# ── Cached vector store builder (avoids re-embedding on every rerun) ────────
@st.cache_resource(ttl=1800, show_spinner=False)
def get_cached_vector_store(username, repo_names_key):
    """repo_names_key is a stable string derived from repo names, used only
    to invalidate cache if the repo set changes without changing the code."""
    repo_info, err = fetch_user_repos(username, include_topics=st.session_state.get("include_topics", False),
                                       max_repos=st.session_state.get("max_repos", 30))
    if err:
        return None, err, None
    vector_store, err = build_vector_store(repo_info)
    return vector_store, err, repo_info


# ── LLM ───────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_llm():
    # Model name now comes from GEMINI_MODEL in .env (see top of file) so a
    # future deprecation only requires updating .env, not this code.
    return GoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.2,
        transport="rest",
    )


# ── Single-call RAG answer (retrieval + one LLM call, not two) ──────────────
def answer_question(vector_store, question, chat_history, k=4):
    docs = vector_store.similarity_search(question, k=k)
    context = "\n\n".join(d.page_content for d in docs)

    history_text = ""
    for turn in chat_history[-6:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_text += f"{role}: {turn['content']}\n"

    prompt = f"""You are analyzing a GitHub user's repositories based on the context below.
Answer the question clearly and concisely. Reference repository names and links where relevant.

Context (repository data):
{context}

Conversation so far:
{history_text}

Question: {question}
Answer:"""

    llm = get_llm()
    answer = call_with_retry(llm.invoke, prompt)
    tokens = estimate_tokens(prompt) + estimate_tokens(answer)
    return answer, tokens


def analyze_most_complex(vector_store, username):
    query = f"most technically challenging complex repository for {username}"
    docs = vector_store.similarity_search(query, k=min(6, len(vector_store.docstore._dict)))
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""Based on the repository data below for GitHub user '{username}', identify the
single most technically challenging repository. Consider languages used, stars, forks,
number of open issues, topics, and description.

Repository data:
{context}

Respond in this exact format:
Repository Name: <name>
Repository Link: [<name>](<url>)
Analysis: <detailed explanation of why it is the most complex>
"""
    llm = get_llm()
    start_time = time.time()
    answer = call_with_retry(llm.invoke, prompt)
    elapsed = round(time.time() - start_time, 2)
    tokens = estimate_tokens(prompt) + estimate_tokens(answer)
    return answer, elapsed, tokens


# ── Main app ─────────────────────────────────────────────────────────────────
def main():
    config.init()

    st.title("🔍 GitHub Repo Analyzer")
    st.sidebar.title("GitHub Repo Analyzer")

    # ── Hard stop if required keys are missing ───────────────────────────────
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not GITHUB_TOKEN:
        missing.append("GITHUB_TOKEN")
    if missing:
        st.error(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Add them to your .env file and restart the app."
        )
        st.stop()

    # ── Session state ─────────────────────────────────────────────────────────
    for key, default in [
        ("chat_history", []),
        ("vector_store", None),
        ("analyzed", False),
        ("response_time", 0.0),
        ("repos_analyzed", 0),
        ("total_tokens", 0),
        ("include_topics", False),
        ("max_repos", 30),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Sidebar ──────────────────────────────────────────────────────────────
    username = st.sidebar.text_input("Enter GitHub Username")
    st.session_state.max_repos = st.sidebar.slider(
        "Max repos to analyze", min_value=5, max_value=100, value=st.session_state.max_repos, step=5,
        help="Lower this to reduce Gemini API usage."
    )
    st.session_state.include_topics = st.sidebar.checkbox(
        "Include repo topics (extra GitHub API calls)", value=st.session_state.include_topics
    )
    submit_button = st.sidebar.button("Submit")

    st.sidebar.header("About")
    st.sidebar.info(
        "This tool analyzes a GitHub user's repositories using Gemini AI. "
        "It finds the most technically complex repo and lets you ask follow-up questions. "
        "Uses batched, rate-limited requests to stay within the Gemini free tier."
    )

    st.divider()

    # ── Reset session on new submission ──────────────────────────────────────
    if submit_button and username:
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.analyzed = False
        st.session_state.response_time = 0.0
        st.session_state.repos_analyzed = 0
        st.session_state.total_tokens = 0

    # ── Main analysis ────────────────────────────────────────────────────────
    if submit_button and username:
        with st.spinner("Fetching repositories from GitHub..."):
            repo_info, err = fetch_user_repos(
                username,
                include_topics=st.session_state.include_topics,
                max_repos=st.session_state.max_repos,
            )

        if err:
            st.error(err)
            st.stop()

        if not repo_info:
            st.warning(f"No public non-fork repositories found for `{username}`.")
            st.stop()

        st.subheader(f"📁 Repositories for `{username}` ({len(repo_info)} shown)")
        display_repos(repo_info)
        st.session_state.repos_analyzed = len(repo_info)

        st.info("⚙️ Building knowledge base using Gemini embeddings... please wait.")
        repo_names_key = "|".join(r["name"] for r in repo_info)
        with st.spinner("Embedding repository data..."):
            try:
                vector_store, vs_err, _ = get_cached_vector_store(username, repo_names_key)
            except Exception as e:
                st.error(f"Failed to build knowledge base: {e}")
                st.stop()

        if vs_err:
            st.error(vs_err)
            st.stop()

        st.session_state.vector_store = vector_store

        st.subheader("🏆 Most Technically Complex Repository")
        with st.spinner("Analyzing with Gemini AI..."):
            try:
                analysis, elapsed, tokens = analyze_most_complex(vector_store, username)
            except Exception as e:
                if "404" in str(e) and "no longer available" in str(e).lower():
                    st.error(
                        f"The model '{GEMINI_MODEL}' has been deprecated by Google. "
                        f"Update GEMINI_MODEL in your .env file to a current model name "
                        f"(check https://ai.google.dev/gemini-api/docs/models) and restart the app."
                    )
                else:
                    st.error(f"Gemini API error while analyzing: {e}")
                st.stop()

            st.markdown(analysis)
            st.session_state.chat_history.append({"role": "assistant", "content": analysis})
            st.session_state.response_time = elapsed
            st.session_state.total_tokens += tokens
            st.session_state.analyzed = True

    # ── KPI Dashboard ────────────────────────────────────────────────────────
    if st.session_state.analyzed:
        st.divider()
        st.subheader("📊 Performance Metrics")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Response Time", f"{st.session_state.response_time}s",
                       help="Time taken by Gemini AI to analyze repositories")
        with col2:
            st.metric("📁 Repos Analyzed", st.session_state.repos_analyzed,
                       help="Total non-fork repositories found and analyzed")
        with col3:
            st.metric("🔢 Tokens Used (est.)", f"{st.session_state.total_tokens:,}",
                       help="Estimated tokens consumed in this session")

    # ── Q&A Chat Section ─────────────────────────────────────────────────────
    if st.session_state.vector_store and st.session_state.analyzed:
        st.divider()
        st.subheader("💬 Ask Anything About These Repositories")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask a question about the repositories...")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer, tokens = answer_question(
                            st.session_state.vector_store, user_input, st.session_state.chat_history
                        )
                        st.session_state.total_tokens += tokens
                    except Exception as e:
                        answer = f"⚠️ Gemini API error: {e}"
                    st.markdown(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()