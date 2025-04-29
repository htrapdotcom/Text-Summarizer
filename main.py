import io
import tempfile
import os
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from fpdf import FPDF
from streamlit_player import st_player
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
import requests
import pyperclip
import google.generativeai as genai
import openai
from openai import Client


# Load environment variables
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")

# Streamlit app title
st.title("YouTube Video Summarizer and Chatbot")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    api_choice = st.selectbox("Choose API:", ["Google Gemini", "OpenAI"])

# Function to fetch video metadata
def get_video_details(video_id):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={youtube_api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "items" in data and len(data["items"]) > 0:
            snippet = data["items"][0]["snippet"]
            stats = data["items"][0]["statistics"]
            return {
                "title": snippet["title"],
                "channel": snippet["channelTitle"],
                "views": stats.get("viewCount", "N/A"),
                "likes": stats.get("likeCount", "N/A"),
                "comments": stats.get("commentCount", "N/A"),
            }
    return None

# Helper Functions

def fetch_transcript(video_url):
    try:
        video_id = video_url.split("=")[1]
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([item["text"] for item in transcript_data])
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

def correct_transcript_with_gemini(transcript, api_key):
    """Correct the transcript using Google Gemini."""
    correction_prompt = """
    You are an assistant that helps correct transcripts by fixing punctuation, grammar, and spelling errors. 
    Please improve the following transcript:

    Transcript: {transcript}
    """
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel("gemini-pro")
        corrected_response = model.generate_content(
            correction_prompt.format(transcript=transcript)
        )
        return corrected_response.text
    except Exception as e:
        return str(e)

def correct_transcript_with_openai(transcript, api_key):
    """Correct the transcript using OpenAI."""
    openai.api_key = api_key

    # Split the transcript into chunks to ensure token limit compliance
    max_chunk_size = 2048  # Approximate max token size per API call
    transcript_chunks = [
        transcript[i : i + max_chunk_size]
        for i in range(0, len(transcript), max_chunk_size)
    ]
    
    corrected_transcript = []
    try:
        client = Client()
        for chunk in transcript_chunks:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that corrects grammar, punctuation, and spelling in transcripts.",
                    },
                    {"role": "user", "content": chunk},
                ],
            )
            corrected_transcript.append(response.choices[0].message.content)
        return "\n".join(corrected_transcript)
    except Exception as e:
        return str(e)

def correct_transcript(transcript, api_choice, api_key):
    """Correct the transcript using the selected API."""
    if api_choice == "Google Gemini":
        return correct_transcript_with_gemini(transcript, api_key)
    elif api_choice == "OpenAI":
        return correct_transcript_with_openai(transcript, api_key)
    else:
        raise ValueError("Invalid API choice!")
    
def generate_summary_with_gemini(transcript, api_key):
    """Generate a summary using Google Gemini."""
    summary_prompt = """
    You are a YouTube video summarizer. Summarize the transcript text in points within 250 words:
    """
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(summary_prompt + transcript)
        return response.text
    except Exception as e:
        return str(e)

def generate_summary_with_openai(transcript, api_key):
    """Generate a summary using OpenAI."""
    openai.api_key = api_key
    try:
        client = Client()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the transcript text in points within 250 words.",
                },
                {"role": "user", "content": transcript},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

def generate_summary(transcript, api_choice, api_key):
    """Generate a summary using the selected API."""
    if api_choice == "Google Gemini":
        return generate_summary_with_gemini(transcript, api_key)
    elif api_choice == "OpenAI":
        return generate_summary_with_openai(transcript, api_key)
    else:
        raise ValueError("Invalid API choice!")

def create_embeddings(chunks, api_choice, api_key):
    if api_choice == "Google Gemini":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", api_key=api_key)
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
    return FAISS.from_texts(chunks, embedding=embeddings)

def run_chain(question, vector_store, api_choice, api_key):
    chain = load_qa_chain(
        ChatGoogleGenerativeAI(api_key=api_key) if api_choice == "Google Gemini" else ChatOpenAI(api_key=api_key),
        prompt=PromptTemplate(template="""Context: {context}\n\nQuestion: {question}\n\nAnswer:""", input_variables=["context", "question"]),
    )
    return chain.run({"input_documents": vector_store.similarity_search(question), "question": question})


# New chatbot functions
def generate_response_with_gemini(transcript, question, api_key):
    """Generate a response using Google Gemini."""
    chatbot_prompt = """
    You are a helpful assistant that answers questions about video content.
    Use the following transcript to answer the question:
    
    Transcript: {transcript}
    
    Question: {question}
    
    Please provide a clear and concise answer based on the transcript content.
    """
    
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(
            chatbot_prompt.format(transcript=transcript, question=question)
        )
        return response.text
    except Exception as e:
        return str(e)

def generate_response_with_openai(transcript, question, api_key):
    """Generate a response using OpenAI."""
    openai.api_key = api_key
    try:
        client = Client()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions about video content. Use the transcript to answer questions accurately.",
                },
                {
                    "role": "user",
                    "content": f"Transcript: {transcript}\n\nQuestion: {question}"
                },
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

def generate_chatbot_response(transcript, question, api_choice, api_key):
    """Generate a chatbot response using the selected API."""
    if api_choice == "Google Gemini":
        return generate_response_with_gemini(transcript, question, api_key)
    elif api_choice == "OpenAI":
        return generate_response_with_openai(transcript, question, api_key)
    else:
        raise ValueError("Invalid API choice!")

def generate_pdf_content(title, content_selections):
    """Generate PDF content based on selected content options"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Video Title: {title}", ln=1, align='C')
    pdf.ln(10)

    for selection in content_selections:
        content = selection.get('content')
        header = selection.get('header')
        
        if content:
            pdf.cell(0, 10, txt=header, ln=1)
            pdf.ln(5)
            
            if isinstance(content, list):  # For summaries or chat history
                for idx, item in enumerate(content, 1):
                    if isinstance(item, dict):  # Chat history
                        pdf.multi_cell(0, 10, txt=f"Q{idx}: {item['question']}")
                        pdf.multi_cell(0, 10, txt=f"A{idx}: {item['answer']}")
                    else:  # Summaries
                        pdf.multi_cell(0, 10, txt=f"Summary {idx}:\n{item}")
                    pdf.ln(5)
            else:  # For transcripts
                pdf.multi_cell(0, 10, txt=content)
            pdf.ln(10)

    return pdf.output(dest='S').encode('latin1')

def generate_text_content(title, content_selections):
    """Generate text content based on selected content options"""
    content = [f"Video Title: {title}\n"]
    
    for selection in content_selections:
        content_data = selection.get('content')
        header = selection.get('header')
        
        if content_data:
            content.append(f"\n{header}")
            
            if isinstance(content_data, list):  # For summaries or chat history
                for idx, item in enumerate(content_data, 1):
                    if isinstance(item, dict):  # Chat history
                        content.append(f"\nQ{idx}: {item['question']}")
                        content.append(f"A{idx}: {item['answer']}")
                    else:  # Summaries
                        content.append(f"\nSummary {idx}:")
                        content.append(item)
            else:  # For transcripts
                content.append(content_data)
    
    return "\n".join(content)

# Main Workflow
youtube_url = st.text_input("Enter YouTube Video URL:")

if youtube_url:
    video_id = youtube_url.split("=")[1]
    st_player(f"https://www.youtube.com/watch?v={video_id}")

    video_details = get_video_details(video_id)
    if video_details:
        st.markdown(f"**Title:** {video_details['title']}")
        st.markdown(f"**Channel:** {video_details['channel']}")
        st.markdown(f"**Views:** {video_details['views']}")
        st.markdown(f"**Likes:** {video_details['likes']}")

    # Fetch Transcript
    if st.button("Fetch Transcript"):
        transcript = fetch_transcript(youtube_url)
        if transcript:
            st.session_state["transcript"] = transcript
            st.success("Transcript fetched successfully!")

    # Display Raw Transcript
    if "transcript" in st.session_state:
        with st.expander("Raw Transcript", expanded=False):
            st.text(st.session_state["transcript"])

    # Correct Transcript
    if "transcript" in st.session_state:
        if st.button("Generate AI-Corrected Transcript"):
            try:
                corrected = correct_transcript(
                    st.session_state["transcript"],
                    api_choice,
                    google_api_key if api_choice == "Google Gemini" else openai_api_key,
                )
                st.session_state["corrected_transcript"] = corrected
                st.success("AI-Corrected Transcript generated successfully!")
            except Exception as e:
                st.error(f"Error during correction: {e}")

    # Display AI-Corrected Transcript
    if "corrected_transcript" in st.session_state:
        with st.expander("AI-Corrected Transcript", expanded=False):
            st.text(st.session_state["corrected_transcript"])

    # Summarize Transcript
    if "transcript" in st.session_state:
        if st.button("Summarize"):
            try:
                summary = generate_summary(
                    st.session_state["transcript"],
                    api_choice,
                    google_api_key if api_choice == "Google Gemini" else openai_api_key,
                )
                st.session_state["latest_summary"] = summary
                if summary not in st.session_state.get("summaries", []):
                    st.session_state.setdefault("summaries", []).append(summary)
                st.success("Summary generated successfully!")
                st.write(st.session_state["latest_summary"])
            except Exception as e:
                st.error(f"Error during summarization: {e}")

    # Dropdown for Summaries
    if st.session_state.get("summaries"):
        selected_summary = st.selectbox(
            "Select a Summary",
            options=st.session_state["summaries"],
            format_func=lambda x: f"Summary {st.session_state['summaries'].index(x) + 1}",
        )
        st.write(selected_summary)
    

    # Initialize session state for chat_history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Chatbot Interface
    st.markdown("### Chatbot")
    question = st.text_input("Ask a question about the video:")

    if question and "transcript" in st.session_state:
        # Check if the question already exists in chat history
        existing_answer = None
        for chat in st.session_state["chat_history"]:
            if chat["question"] == question:
                existing_answer = chat["answer"]
                break

        if not existing_answer:
            try:
                response_text = generate_chatbot_response(
                    st.session_state["transcript"],
                    question,
                    api_choice,
                    google_api_key if api_choice == "Google Gemini" else openai_api_key,
                )
                st.session_state["chat_history"].append({"question": question, "answer": response_text})
                existing_answer = response_text
            except Exception as e:
                st.error(f"Error generating response: {e}")
                existing_answer = "Sorry, I encountered an error while generating the response."

        st.markdown(f"**Answer:** {existing_answer}")

        # Copy options for current response
        with st.expander("Copy Options"):
            copy_option = st.radio(
                "Select text to copy:",
                ("Answer only", "Question and Answer"),
                key="copy_option_chat"
            )
            if st.button("Copy Text", key="copy_button_chat"):
                if copy_option == "Answer only":
                    pyperclip.copy(existing_answer)
                    st.success("Answer copied to clipboard!")
                else:
                    combined_text = f"Q: {question}\nA: {existing_answer}"
                    pyperclip.copy(combined_text)
                    st.success("Question and Answer copied to clipboard!")

    # Chat History
    if st.session_state["chat_history"]:
        st.markdown("### Chat History")
        for i, chat in enumerate(st.session_state["chat_history"]):
            with st.expander(f"Q{i+1}: {chat['question'][:50]}..."):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                
                copy_option = st.radio(
                    "Select text to copy:",
                    ("Answer only", "Question and Answer"),
                    key=f"copy_option_{i}"
                )
                if st.button("Copy Text", key=f"copy_button_{i}"):
                    if copy_option == "Answer only":
                        pyperclip.copy(chat['answer'])
                        st.success("Answer copied to clipboard!")
                    else:
                        combined_text = f"Q: {chat['question']}\nA: {chat['answer']}"
                        pyperclip.copy(combined_text)
                        st.success("Question and Answer copied to clipboard!")

        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state["chat_history"] = []
            st.experimental_rerun()

        # Export options
        if "transcript" in st.session_state:
            st.markdown("### Export Options")
            
            # Export content selection
            st.subheader("Select Content to Export")
            
            export_options = {
                "raw_transcript": st.checkbox("Raw Transcript"),
                "corrected_transcript": st.checkbox("AI-Corrected Transcript"),
                "all_summaries": st.checkbox("All Summaries"),
                "chat_history": st.checkbox("Chat History"),
            }
            
            # Only show specific summary selection if there are summaries
            specific_summary = None
            if st.session_state.get("summaries"):
                show_specific_summary = st.checkbox("Select Specific Summary")
                if show_specific_summary:
                    summary_options = [f"Summary {i+1}" for i in range(len(st.session_state["summaries"]))]
                    specific_summary = st.selectbox("Choose Summary", summary_options)
            
            # Export buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Generate PDF"):
                    if not any(export_options.values()) and not specific_summary:
                        st.warning("Please select at least one content option to export.")
                    else:
                        # Get title
                        title = st.session_state.get("video_details", {}).get("title", "Video_Transcript")
                        
                        # Prepare content selections based on user choices
                        content_selections = []
                        
                        if export_options["raw_transcript"]:
                            content_selections.append({
                                'header': 'Raw Transcript',
                                'content': st.session_state.get("transcript")
                            })
                        
                        if export_options["corrected_transcript"]:
                            content_selections.append({
                                'header': 'AI-Corrected Transcript',
                                'content': st.session_state.get("corrected_transcript")
                            })
                        
                        if export_options["all_summaries"]:
                            content_selections.append({
                                'header': 'Summaries',
                                'content': st.session_state.get("summaries", [])
                            })
                        
                        if specific_summary and not export_options["all_summaries"]:
                            summary_idx = int(specific_summary.split()[-1]) - 1
                            content_selections.append({
                                'header': f'Summary {summary_idx + 1}',
                                'content': [st.session_state["summaries"][summary_idx]]
                            })
                        
                        if export_options["chat_history"]:
                            content_selections.append({
                                'header': 'Chat History',
                                'content': st.session_state.get("chat_history", [])
                            })
                        
                        # Generate PDF content
                        pdf_content = generate_pdf_content(title, content_selections)
                        
                        # Store in session state for download
                        st.session_state["pdf_content"] = pdf_content
                        st.session_state["pdf_filename"] = f"{title.replace(' ', '_')}_export.pdf"
                        st.success("PDF generated! Click the download button to save it.")
                
                if "pdf_content" in st.session_state:
                    st.download_button(
                        label="Download PDF",
                        data=st.session_state["pdf_content"],
                        file_name=st.session_state["pdf_filename"],
                        mime="application/pdf"
                    )
            
            with col2:
                if st.button("Generate Text File"):
                    if not any(export_options.values()) and not specific_summary:
                        st.warning("Please select at least one content option to export.")
                    else:
                        # Get title
                        title = st.session_state.get("video_details", {}).get("title", "Video_Transcript")
                        
                        # Prepare content selections based on user choices
                        content_selections = []
                        
                        if export_options["raw_transcript"]:
                            content_selections.append({
                                'header': 'Raw Transcript',
                                'content': st.session_state.get("transcript")
                            })
                        
                        if export_options["corrected_transcript"]:
                            content_selections.append({
                                'header': 'AI-Corrected Transcript',
                                'content': st.session_state.get("corrected_transcript")
                            })
                        
                        if export_options["all_summaries"]:
                            content_selections.append({
                                'header': 'Summaries',
                                'content': st.session_state.get("summaries", [])
                            })
                        
                        if specific_summary and not export_options["all_summaries"]:
                            summary_idx = int(specific_summary.split()[-1]) - 1
                            content_selections.append({
                                'header': f'Summary {summary_idx + 1}',
                                'content': [st.session_state["summaries"][summary_idx]]
                            })
                        
                        if export_options["chat_history"]:
                            content_selections.append({
                                'header': 'Chat History',
                                'content': st.session_state.get("chat_history", [])
                            })
                        
                        # Generate text content
                        text_content = generate_text_content(title, content_selections)
                        
                        # Store in session state for download
                        st.session_state["text_content"] = text_content
                        st.session_state["text_filename"] = f"{title.replace(' ', '_')}_export.txt"
                        st.success("Text file generated! Click the download button to save it.")
                
                if "text_content" in st.session_state:
                    st.download_button(
                        label="Download Text",
                        data=st.session_state["text_content"],
                        file_name=st.session_state["text_filename"],
                        mime="text/plain"
                    )