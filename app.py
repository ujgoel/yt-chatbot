"""Chat with a YouTube video: transcript -> chunks -> embeddings -> FAISS -> Groq LLM."""
import os
import re
import ssl
from urllib.parse import parse_qs, urlparse

import groq
import httpx
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

LLM_MODEL = "openai/gpt-oss-20b"  # fixed on purpose: visitors cannot pick the model
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # supports Hindi
PREFERRED_LANGS = ["hi", "en"]
MAX_TRANSCRIPT_CHARS = 300_000
MAX_QUESTION_CHARS = 500
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

PROMPT = PromptTemplate(
    template="""You are an experienced educator.
Your goal is to answer the user query based on the given context only, strictly no hallucination. If you can't answer something from the context then simply say 'I don't know'.
Even if the context is in Hindi, translate it into English and answer the query in English only.

context: {context}

query: {query}
""",
    input_variables=["context", "query"],
)


# ---------- helpers ----------
def get_api_key() -> str | None:
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GROQ_API_KEY")  # Streamlit Cloud secrets
    except Exception:
        return None


def extract_video_id(url: str) -> str | None:
    url = url.strip()
    if re.fullmatch(r"[\w-]{11}", url):
        return url
    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    video_id = None
    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
    elif host in ("youtube.com", "music.youtube.com", "youtube-nocookie.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        else:
            m = re.match(r"^/(?:embed|shorts|live|v)/([\w-]{11})", parsed.path)
            video_id = m.group(1) if m else None
    return video_id if video_id and re.fullmatch(r"[\w-]{11}", video_id) else None


def is_tls_error(exc: BaseException) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (ssl.SSLError, requests.exceptions.SSLError)):
            return True
        if "CERTIFICATE_VERIFY_FAILED" in str(exc) or "SSL: " in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


@st.cache_resource
def tls_state() -> dict:
    """Process-wide flag: flips to insecure only after a TLS failure was seen."""
    return {"insecure": False}


# ---------- pipeline ----------
def fetch_transcript(video_id: str) -> str:
    def _fetch(api: YouTubeTranscriptApi) -> str:
        transcripts = api.list(video_id)
        try:
            chosen = transcripts.find_transcript(PREFERRED_LANGS)
        except Exception:
            chosen = next(iter(transcripts))  # any available language
        return " ".join(chunk.text for chunk in chosen.fetch())

    try:
        return _fetch(YouTubeTranscriptApi())
    except Exception as e:
        if not is_tls_error(e):
            raise
        session = requests.Session()
        session.verify = False
        return _fetch(YouTubeTranscriptApi(http_client=session))


@st.cache_resource(show_spinner=False)
def get_embeddings() -> FastEmbedEmbeddings:
    # The model is downloaded from Hugging Face on first use. If TLS verification fails
    # (e.g. corporate proxy), silently fall back to an unverified client for the download.
    try:
        httpx.get("https://huggingface.co", timeout=10)
    except Exception as e:
        if is_tls_error(e):
            try:
                from huggingface_hub import set_client_factory

                set_client_factory(lambda: httpx.Client(verify=False, follow_redirects=True, timeout=None))
            except ImportError:  # older huggingface_hub uses requests
                from huggingface_hub import configure_http_backend

                def _factory() -> requests.Session:
                    s = requests.Session()
                    s.verify = False
                    return s

                configure_http_backend(backend_factory=_factory)
    return FastEmbedEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource(show_spinner=False, max_entries=5)
def build_index(video_id: str) -> FAISS:
    """One separate in-memory FAISS index per video id, so videos never mix."""
    transcript = fetch_transcript(video_id)[:MAX_TRANSCRIPT_CHARS]
    if not transcript.strip():
        raise ValueError("The transcript is empty.")
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_text(transcript)
    return FAISS.from_texts(chunks, get_embeddings())


def get_llm(insecure: bool, api_key: str) -> ChatGroq:
    # Not cached on purpose: user-supplied keys must not be kept in a shared server cache.
    kwargs = {"http_client": httpx.Client(verify=False)} if insecure else {}
    return ChatGroq(model=LLM_MODEL, api_key=api_key, **kwargs)


def check_groq_key(key: str) -> str:
    """Return 'valid', 'invalid' (rejected by Groq) or 'unknown' (could not verify)."""

    def _get(verify: bool) -> httpx.Response:
        return httpx.get(GROQ_MODELS_URL, headers={"Authorization": f"Bearer {key}"}, timeout=10, verify=verify)

    try:
        try:
            resp = _get(True)
        except Exception as e:
            if not is_tls_error(e):
                raise
            resp = _get(False)
    except Exception:
        return "unknown"
    if resp.status_code == 200:
        return "valid"
    if resp.status_code == 401:
        return "invalid"
    return "unknown"


def format_context(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def ask(question: str, index: FAISS, api_key: str) -> str:
    retriever = index.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    state = tls_state()
    for attempt in range(2):
        chain = (
            RunnableParallel(
                {"context": retriever | RunnableLambda(format_context), "query": RunnablePassthrough()}
            )
            | PROMPT
            | get_llm(state["insecure"], api_key)
            | StrOutputParser()
        )
        try:
            return chain.invoke(question)
        except Exception as e:
            if attempt == 0 and not state["insecure"] and is_tls_error(e):
                state["insecure"] = True  # silent fallback, then retry once
                continue
            raise
    raise RuntimeError("unreachable")


# ---------- UI ----------
st.set_page_config(page_title="YouTube Chatbot", page_icon="🎬")
st.title("🎬 Chat with a YouTube video")

st.session_state.setdefault("video_id", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("key_check", ("", ""))  # (key, status) of the last validated user key

default_key = get_api_key()
user_key = ""
key_error = None  # set when chatting must be blocked; holds the message to show

with st.sidebar:
    st.header("API key (optional)")
    user_key = st.text_input(
        "Your Groq API key",
        type="password",
        placeholder="gsk_...",
        help="Leave empty to use the app's default key. Your key is only kept for this session.",
    ).strip()

    if user_key:
        if st.session_state.key_check[0] != user_key or st.session_state.key_check[1] == "unknown":
            with st.spinner("Checking your API key..."):
                st.session_state.key_check = (user_key, check_groq_key(user_key))
        status = st.session_state.key_check[1]
        if status == "valid":
            st.success("Using your API key.")
        elif status == "invalid":
            key_error = "Your API key is not valid. Remove it to use the default key."
        else:
            key_error = "Couldn't verify your API key right now. Try again, or remove it to use the default key."
    elif default_key:
        st.caption("Using the app's default key.")
    else:
        key_error = "The app has no default API key configured. Please enter your own Groq API key."

    if key_error:
        st.error(key_error)

    st.divider()
    st.header("Video")
    with st.form("load_video"):
        url = st.text_input("YouTube link", placeholder="https://www.youtube.com/watch?v=...")
        load = st.form_submit_button("Load video")

    if load:
        # Always drop the previous video and chat first, so a failed load can never
        # leave answers coming from an old video.
        st.session_state.video_id = None
        st.session_state.messages = []
        vid = extract_video_id(url)
        if not vid:
            st.error("That doesn't look like a valid YouTube link.")
        else:
            try:
                with st.spinner("Fetching transcript and indexing (first run downloads the embedding model)..."):
                    build_index(vid)
                st.session_state.video_id = vid
            except Exception as e:
                name = type(e).__name__
                if name in ("TranscriptsDisabled", "NoTranscriptFound", "VideoUnavailable", "StopIteration"):
                    st.error("No transcript is available for this video.")
                else:
                    st.error("Could not load this video. YouTube may be blocking the request; try again later or try another video.")

    if st.session_state.video_id:
        st.success("Active video:")
        st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")
    else:
        st.info("Paste a link and click **Load video**.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.video_id:
    st.chat_input("Load a video first", disabled=True)
elif key_error:
    st.chat_input("Fix the API key in the sidebar first", disabled=True)
elif question := st.chat_input("Ask something about the video"):
    question = question.strip()[:MAX_QUESTION_CHARS]
    vid = st.session_state.video_id
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer = ask(question, build_index(vid), user_key or default_key)
        except groq.AuthenticationError:
            if user_key:  # key was valid earlier but has since been revoked
                st.session_state.key_check = (user_key, "invalid")
                answer = "Your API key is not valid. Remove it to use the default key."
            else:
                answer = "Sorry, the service is unavailable right now. Please try again later."
        except Exception:
            answer = "Sorry, I couldn't get an answer right now (the service may be busy). Please try again in a moment."
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
