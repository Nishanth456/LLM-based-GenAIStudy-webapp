import streamlit as st
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import Replicate
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

os.environ["REPLICATE_API_TOKEN"] = "YOUR_API_TOKEN"

load_dotenv()

# Prompt template for generating 5 MCQs from a PDF
prompt_template_mcq = """
You are an expert in creating practice questions based on study material.
Your goal is to prepare a student for their exam. You do this by asking questions about the text below:

------------
{text}
------------

Create 5 multiple-choice questions that will prepare the student for their exam. Provide options (A, B, C, D) for each question.

QUESTIONS:

"""

# Create the PromptTemplate for generating 5 MCQs
PROMPT_MCQ = PromptTemplate(template=prompt_template_mcq, input_variables=["text"])

# Prompt template for refining questions with additional context
refine_template_mcq = """
You are an expert in creating practice questions based on study material.
Your goal is to help a student test their knowledge on the study material.
We have received some practice questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones (only if necessary) with some more context below.
------------
{text}
------------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.

QUESTIONS:
"""

# Create the PromptTemplate for refining 5 MCQs
REFINE_PROMPT_MCQ = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_mcq,
)

# Initialize Streamlit app
st.title('MCQ Generator:books:')
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
st.subheader('Built for Professionals, Teachers, Students')
st.markdown('<style>h3{color: pink;  text-align: center;}</style>', unsafe_allow_html=True)

# File upload widget
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# Set file path
file_path = None

# Check if a file is uploaded
if uploaded_file:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name

# Check if file_path is set
if file_path:
    # Load data from the uploaded PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # Combine text from Document into one string for MCQ generation
    text_mcq_gen = ''
    for page in data:
        text_mcq_gen += page.page_content

    # Initialize Text Splitter for MCQ generation
    text_splitter_mcq_gen = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=50)

    # Split text into chunks for MCQ generation
    text_chunks_mcq_gen = text_splitter_mcq_gen.split_text(text_mcq_gen)

    # Convert chunks into Documents for MCQ generation
    docs_mcq_gen = [Document(page_content=t) for t in text_chunks_mcq_gen]

    # Initialize Large Language Model for MCQ generation
    llm_mcq_gen = Replicate(
        model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
        input={"temperature": 0.01, "max_length": 500, "top_p": 1}
    )

    # Initialize MCQ generation chain
    mcq_gen_chain = load_summarize_chain(llm=llm_mcq_gen, chain_type="refine", verbose=True, question_prompt=PROMPT_MCQ, refine_prompt=REFINE_PROMPT_MCQ)

    # Run MCQ generation chain
    generated_mcq = mcq_gen_chain.run(docs_mcq_gen)

    # Display MCQs in Streamlit
    st.subheader('Generated Multiple-Choice Questions:')

    # Always treat generated_mcq as a string
    # Split the string into lines
    lines = generated_mcq.split('\n')

    st.write(f"Generated MCQs: {generated_mcq}")

# Cleanup temporary files
if file_path:
    os.remove(file_path)
