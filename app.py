import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import requests

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# YouTube Data API key
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Prompt for Google Gemini AI
prompt = """
You are a YouTube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points
within 250 words. Please provide the summary of the text given here:  
"""

# Function to get transcript details from YouTube video
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        raise e

# Function to get video metadata (title and channel name) using YouTube Data API
def get_video_details(video_id):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={YOUTUBE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "items" in data and len(data["items"]) > 0:
            video_title = data["items"][0]["snippet"]["title"]
            channel_name = data["items"][0]["snippet"]["channelTitle"]
            return video_title, channel_name
        else:
            return None, None
    else:
        raise Exception("Error fetching video details")

# Function to generate a summary using Google Gemini AI
def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

# Streamlit app interface
st.title("YouTube Video Summarizer")
youtube_link = st.text_input("YouTube Video Link:")

if youtube_link:
    video_id = youtube_link.split("=")[1]
    
    # Display thumbnail
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
    
    # Fetch and display video title and channel name
    try:
        video_title, channel_name = get_video_details(video_id)
        if video_title and channel_name:
            st.markdown(f"**Video Title:** {video_title}")
            st.markdown(f"**Channel:** {channel_name}")
        else:
            st.warning("Could not fetch video details.")
    except Exception as e:
        st.error(f"Error fetching video details: {e}")

# Button to get detailed notes
if st.button("Summarize"):
    transcript_text = extract_transcript_details(youtube_link)
    
    if transcript_text:
        summary = generate_gemini_content(transcript_text, prompt)
        st.write(summary)
