import streamlit as st
import requests
import os
import time
from dotenv import load_dotenv
import pandas as pd
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from github import Github
import utils.config as config
from utils.constants import *

# ── Load env variables ───────────────────────────────────────────────────────
load_dotenv()
if os.getenv('GEMINI_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API_KEY')
if os.getenv('GITHUB_TOKEN'):
    os.environ['GITHUB_TOKEN'] = os.getenv('GITHUB_TOKEN')

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
        /* KPI metric cards */
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
    # Rough estimate: 1 token ≈ 4 characters
    return len(str(text)) // 4


# ── Fetch GitHub repos via REST API ─────────────────────────────────────────
@st.cache_data
def fetch_github_repos(username):
    repos = []
    page = 1
    while True:
        url = f"https://api.github.com/users/{username}/repos?page={page}&per_page=50"
        response = requests.get(url, headers={"Authorization": f"token {os.getenv('GITHUB_TOKEN')}"})
        if response.status_code == 404:
            st.error("Invalid username. Please try again.")
            st.stop()
        if response.status_code == 403:
            st.error("API rate limit exceeded. Please try again later.")
            st.stop()
        data = response.json()
        if not data:
            break
        repos.extend(data)
        page += 1
    return repos


# ── Display repos as clickable links ─────────────────────────────────────────
def display_repos(repos):
    for repo in repos:
        repo_name = repo["name"]
        repo_url = repo["html_url"]
        st.write(f"[{repo_name}]({repo_url})")


# ── Build FAISS vector store from GitHub repo data ───────────────────────────
def build_vector_store(username):
    client = Github(os.getenv('GITHUB_TOKEN'))
    user = client.get_user(username)
    repos = user.get_repos()

    repo_info = []
    for repo in repos:
        if repo.fork:
            continue
        repo_info.append({
            "name": repo.name,
            "description": repo.description,
            "language": repo.language,
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "labels": [l.name for l in repo.get_labels()],
            "open_issues": repo.open_issues_count,
            "url": repo.html_url,
        })

    # ── KPI: Repos Analyzed Count ────────────────────────────────────────────
    st.session_state.repos_analyzed = len(repo_info)

    df = pd.DataFrame(repo_info)
    df.to_csv("repo_data.csv", index=False)

    loader = CSVLoader(file_path="repo_data.csv", encoding="utf-8")
    docs = loader.load()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


# ── Build conversational Q&A chain ──────────────────────────────────────────
def build_qa_chain(vector_store):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    llm = GoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        google_api_key=os.getenv('GEMINI_API_KEY')
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=False,
        verbose=False,
    )
    return qa_chain


# ── Auto complexity analysis ─────────────────────────────────────────────────
def analyze_most_complex(qa_chain, username):
    query = f"""
Which is the most technically challenging repository for GitHub user '{username}'?

Consider: languages used, stars, forks, number of issues, labels, and description.

Return in this format:
Repository Name: <name>
Repository Link: https://github.com/{username}/<repo_name>
Analysis: <detailed explanation of why it is the most complex>

Make the repository link clickable: [Repository Name](Repository Link)
"""
    # ── KPI: Response Time - Start ───────────────────────────────────────────
    start_time = time.time()

    result = qa_chain.invoke({"question": query})
    answer = result["answer"]

    # ── KPI: Response Time - End ─────────────────────────────────────────────
    end_time = time.time()
    st.session_state.response_time = round(end_time - start_time, 2)

    # ── KPI: Token Usage ─────────────────────────────────────────────────────
    tokens_used = estimate_tokens(query) + estimate_tokens(answer)
    st.session_state.total_tokens += tokens_used

    return answer


# ── Main app ─────────────────────────────────────────────────────────────────
def main():
    config.init()

    st.title("🔍 GitHub Repo Analyzer")
    st.sidebar.title("GitHub Repo Analyzer")

    # ── Initialize Session State ─────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "analyzed" not in st.session_state:
        st.session_state.analyzed = False

    # ── KPI Session State Init ───────────────────────────────────────────────
    if "response_time" not in st.session_state:
        st.session_state.response_time = 0.0
    if "repos_analyzed" not in st.session_state:
        st.session_state.repos_analyzed = 0
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0

    # ── Sidebar ──────────────────────────────────────────────────────────────
    username = st.sidebar.text_input("Enter GitHub Username")
    submit_button = st.sidebar.button("Submit")

    st.sidebar.header("About")
    st.sidebar.info(
        "This tool analyzes a GitHub user's repositories using Gemini AI "
        "and LangChain. It finds the most technically complex repo and lets "
        "you ask follow-up questions!"
    )

    st.divider()

    # Reset session on new submission
    if submit_button and username:
        st.session_state.chat_history = []
        st.session_state.qa_chain = None
        st.session_state.analyzed = False
        st.session_state.response_time = 0.0
        st.session_state.repos_analyzed = 0
        st.session_state.total_tokens = 0

    # ── Main Analysis ────────────────────────────────────────────────────────
    if submit_button and username:
        st.subheader(f"📁 Repositories for `{username}`")
        repos = fetch_github_repos(username)
        if repos:
            display_repos(repos)

        st.info("⚙️ Building knowledge base using Gemini AI... please wait.")
        vector_store = build_vector_store(username)
        st.session_state.qa_chain = build_qa_chain(vector_store)

        st.subheader("🏆 Most Technically Complex Repository")
        with st.spinner("Analyzing with Gemini AI..."):
            analysis = analyze_most_complex(st.session_state.qa_chain, username)
            st.markdown(analysis)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": analysis
            })
            st.session_state.analyzed = True

    # ── KPI Dashboard ────────────────────────────────────────────────────────
    if st.session_state.analyzed:
        st.divider()
        st.subheader("📊 Performance Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="⏱️ Response Time",
                value=f"{st.session_state.response_time}s",
                help="Time taken by Gemini AI to analyze repositories"
            )
        with col2:
            st.metric(
                label="📁 Repos Analyzed",
                value=st.session_state.repos_analyzed,
                help="Total non-fork repositories found and analyzed"
            )
        with col3:
            st.metric(
                label="🔢 Tokens Used (est.)",
                value=f"{st.session_state.total_tokens:,}",
                help="Estimated tokens consumed in this session"
            )

    # ── Q&A Chat Section ─────────────────────────────────────────────────────
    if st.session_state.qa_chain and st.session_state.analyzed:
        st.divider()
        st.subheader("💬 Ask Anything About These Repositories")

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input("Ask a question about the repositories...")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # ── KPI: Response Time for Q&A ───────────────────────────
                    qa_start = time.time()

                    result = st.session_state.qa_chain.invoke({"question": user_input})
                    answer = result["answer"]

                    # ── KPI: Update Response Time & Tokens ───────────────────
                    st.session_state.response_time = round(time.time() - qa_start, 2)
                    st.session_state.total_tokens += (
                        estimate_tokens(user_input) + estimate_tokens(answer)
                    )

                    st.markdown(answer)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })


if __name__ == "__main__":
    main()