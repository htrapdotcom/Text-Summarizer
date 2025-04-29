# YouTube Video Summarizer and Chatbot

## Description

This Streamlit application provides a comprehensive tool for working with YouTube video content. It allows users to:

- Fetch and display video transcripts.
- Correct transcripts using AI (Google Gemini or OpenAI).
- Generate summaries of videos.
- Engage in a chatbot-style conversation about the video content.
- Export transcripts, summaries, and chat history to PDF or text files.

It leverages the YouTube Transcript API, LangChain, and either Google Gemini or OpenAI for its AI functionalities.

---

## Features

- **Video Information Display:** Shows video title, channel, views, likes, and comments.
- **Transcript Handling:** Fetches, displays, and corrects YouTube video transcripts.
- **AI-Powered Summarization:** Generates video summaries using Google Gemini or OpenAI.
- **Interactive Chatbot:** Allows users to ask questions about the video content.
- **Export Functionality:** Exports selected content (transcripts, summaries, chat history) to PDF or text files.
- **API Flexibility:** Supports both Google Gemini and OpenAI for AI processing.

---

## Requirements

- Python 3.6 or higher
- pip (Python package installer)
- An OpenAI API key (if using OpenAI)
- A Google Gemini API key (if using Gemini)
- A YouTube API key

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Setup

### Clone the Repository

```bash
git clone [repository_url]
cd [repository_directory]
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Obtain API Keys

- Get an **OpenAI API key** from [OpenAI](https://platform.openai.com/).
- Get a **Google Gemini API key** from [Google AI Studio](https://makersuite.google.com/).
- Get a **YouTube API key** from [Google Cloud Console](https://console.cloud.google.com/).

### Set Up Environment Variables

Create a `.env` file in the root directory of the project and add the following lines:

```env
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
YOUTUBE_API_KEY=YOUR_YOUTUBE_API_KEY
```

---

## Usage

Run the Streamlit application:

```bash
streamlit run ai4.py
```

### Workflow

1. Enter a YouTube video URL in the input box.
2. Fetch the transcript.
3. Generate a corrected transcript using the AI model.
4. Generate a summary.
5. Ask questions about the video content in the chatbot interface.
6. Export transcript, summary, or chat history as PDF or text.

---

## Code Explanation

### `ai4.py`

Main Streamlit app script handling:

- UI logic
- API interactions
- Transcript and summary generation
- Chatbot functionalities

### `requirements.txt`

Specifies all necessary Python packages.

### `.env`

(Not included in repository) — Stores API keys securely as environment variables.

---

## FAISS Vector Database Files

### Do Not Push FAISS Files to Git Repository

**Reasons:**

- **Size:** Can bloat the repo with large files.
- **Redundancy:** Easily recreated from transcript.
- **Version Control:** Binary files do not version well.

### Best Practice

- Generate FAISS vector database dynamically during app runtime.
- Creation is handled by the `create_embeddings` function in `ai4.py`.

---

## Future Enhancements

- Support for additional video platforms.
- Enhanced error handling and user feedback.
- More advanced chatbot capabilities.
- Caching mechanisms for better performance.

---

## Author

Parth

---

## License

[MIT License](https://opensource.org/licenses/MIT)&#x20;

