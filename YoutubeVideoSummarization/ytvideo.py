import streamlit as st
from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_API_TOKEN"

# Load OpenAI language model
llm = OpenAI(temperature=0.6)

# Load Streamlit page
st.title("YouTube Video Summarizer 🎥")
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)


# Get user input for YouTube video link
youtube_link = st.text_input("Enter YouTube Video Link:")

# Check if "Submit" button was clicked
submit_clicked = st.button("Summarize")

# Display video metadata if link is provided and "Submit" was clicked
if youtube_link and submit_clicked:
    loader = YoutubeLoader.from_youtube_url(youtube_link, add_video_info=True)
    result = loader.load()
    st.write(f"Found Video from {result[0].metadata['author']} that is {result[0].metadata['length']} seconds long")

    # Split video transcript into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(result)

    # Summarize the video using langchain
    chain = load_summarize_chain(llm=llm, chain_type='map_reduce', verbose=False)
    summary_result = chain.run(texts)

    # Display the summary result
    st.write("Summary:")
    st.write(summary_result)
elif submit_clicked:
    st.write("Please enter a YouTube video link.")
