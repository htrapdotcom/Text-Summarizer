**YouTube Video Summarizer**

A Streamlit-based web application that uses Google Generative AI to summarize YouTube videos. This app retrieves video metadata, transcripts, and generates concise summaries for easy content digestion.

## **Features**

* Extracts transcripts from YouTube videos.  
* Fetches video metadata (title and channel name) using the YouTube Data API.  
* Summarizes video content into key points within 250 words using Google Generative AI.

## **Installation**

Clone the repository:

 git clone https://github.com/your-username/your-repo-name.git  
cd your-repo-name

1. 

Install dependencies:

 pip install \-r requirements.txt

2.   
3. Set up environment variables:

   * Create a `.env` file in the project directory.

Add your **Google API Key** and **YouTube API Key** to the `.env` file:  
 GOOGLE\_API\_KEY="your-google-api-key"  
YOUTUBE\_API\_KEY="your-youtube-api-key"

* 

## **Usage**

Run the Streamlit app:

 streamlit run app.py

1.   
2. Enter the YouTube video link in the provided text box.

3. Click "Summarize" to view the summary and related metadata.

## **Dependencies**

The project requires the following Python libraries:

* `youtube_transcript_api`  
* `streamlit`  
* `google-generativeai`  
* `python-dotenv`

Refer to the `requirements.txt` for all dependencies.

## **Environment Variables**

The `.env` file must contain:

* `GOOGLE_API_KEY`: API key for Google Generative AI.  
* `YOUTUBE_API_KEY`: API key for YouTube Data API.

## **File Structure**

* `app.py`: Main application script.  
* `requirements.txt`: List of dependencies.  
* `.env`: Environment variables (not included in the repository; to be created locally).