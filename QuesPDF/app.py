import streamlit as st
from llama_index import ServiceContext
from llama_index import PromptHelper
from llama_index import VectorStoreIndex
from llama_index.llms import Replicate
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.node_parser import SimpleNodeParser
import os
import PyPDF2
from llama_index.indices.base import Document

# Set Replicate API token
os.environ["REPLICATE_API_TOKEN"] = "YOUR_API_TOKEN"

# Initialize Replicate LLAMA model
llama2_7b_chat = "meta/llama-2-70b-chat"
llm = Replicate(
    model=llama2_7b_chat,
    temperature=0.01,
    additional_kwargs={"top_p": 1, "max_new_tokens": 300}
)

# Initialize Hugging Face Embedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Initialize Node Parser
node_parser = SimpleNodeParser.from_defaults()

prompt_helper = PromptHelper(
  context_window=4096,
  num_output=256,
  chunk_overlap_ratio=0.25,
  chunk_size_limit=None
)

service_context = ServiceContext.from_defaults(
  llm=llm,
  embed_model=embed_model,
  node_parser=node_parser,
  prompt_helper=prompt_helper
)

# Streamlit app
def main():
    st.title('Question your PDF 💬')
    st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
    st.sidebar.title('LLM ChatApp using RAG')
    st.sidebar.markdown('''
    This is an LLM powered application built using:
    - [Streamlit](https://streamlit.io/)
    - [RAG](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)
    - [LLM Model](https://replicate.com/meta/llama-2-70b-chat) 
    ''')

    st.sidebar.write('Developed by - Nishanth')

    # Upload a PDF File
    pdf = st.file_uploader("Upload your PDF File", type='pdf')

    # Create an empty list to store Document objects
    documents = []

    if pdf is not None:
        pdf_reader = PyPDF2.PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Create a Document object with a unique identifier
        document = Document(doc_id=f"unique_id_{len(documents)}", text=text)

        # Append the Document object to the list
        documents.append(document)

        # Create VectorStoreIndex from the list of documents
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        index.storage_context.persist()

        # Query input from the user
        query = st.text_input("Ask a question:")
        query_engine = index.as_query_engine(service_context=service_context)
        if st.button("Submit"):
            if query:
                # Perform query and display results
                response = query_engine.query(query).response
                st.write("Response:", response)
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
