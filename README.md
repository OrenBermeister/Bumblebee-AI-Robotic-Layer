# AI Assistant App

This is a Streamlit-based AI assistant application that integrates multiple AI models including Claude, LLaMA, Gemini, Grok, and DeepSeek. The app provides an interactive interface for engaging with these AI models and maintains conversation history.

## Features

- Multiple AI model support (Claude, LLaMA, Gemini, Grok, DeepSeek)
- Conversation history tracking
- Key learnings extraction
- Audio input/output capabilities
- Beautiful modern UI

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in Streamlit Cloud:
   - OPENAI_API_KEY
   - ANTHROPIC_API_KEY
   - REPLICATE_API_KEY
   - GEMINI_API_KEY
   - TOGETHER_API_KEY

## Running the App

```bash
streamlit run app.py
```

## Deployment

This app is configured for deployment on Streamlit Cloud. Make sure to set all the required environment variables in the Streamlit Cloud settings.
