# YouTube Chatbot

Paste a YouTube link and chat with the video. The transcript is split into chunks, embedded
(multilingual, runs locally on CPU, Hindi supported), stored in an in-memory FAISS index, and
answered by a Groq-hosted LLM using only the retrieved transcript context.

The notebook version of the pipeline lives in `yt-chatbot-src/main.ipynb`; the app is `app.py`.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your Groq key in .env
streamlit run app.py
```

The first video load downloads the embedding model (~220 MB) once.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (`.env` is git-ignored; never commit it).
2. Create the app on [share.streamlit.io](https://share.streamlit.io) with `app.py` as the entry point.
3. In **Settings -> Secrets** add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

The model is fixed in `app.py` (`LLM_MODEL`); visitors cannot change it.

## API key behaviour

- Visitors can optionally enter their own Groq API key in the sidebar.
- Empty field: the app's default key is used.
- Valid key entered: only that key is used.
- Invalid key entered: chat is blocked with a message asking the user to remove the key to use the
  default one (the app never silently falls back to the default key).
- User keys are kept only in the visitor's session, never stored or logged.

## Notes

- Each video gets its own index, and loading a new link clears the chat, so answers never come
  from a previously loaded video.
- If TLS verification fails (e.g. behind a corporate proxy), the app silently retries with
  verification disabled.
- YouTube may block transcript requests from cloud IPs, so some videos can fail when hosted.
