import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
# Set the Google API user input key
if "GOOGLE_API_KEY" not in os.environ:
    google_api_key = st.text_input("Enter your Google API Key:", type="password")
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        st.success("Google API Key has been set successfully!")

def get_single_pdf_chunks(pdf, text_splitter):
    pdf_reader = PdfReader(pdf)
    pdf_chunks = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        page_chunks = text_splitter.split_text(page_text)
        pdf_chunks.extend(page_chunks)
    return pdf_chunks

def get_all_pdfs_chunks(pdf_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=500
    )

    all_chunks = []
    for pdf in pdf_docs:
        pdf_chunks = get_single_pdf_chunks(pdf, text_splitter)
        all_chunks.extend(pdf_chunks)
    
    return all_chunks

def perform_similarity_search(query, collection):
    query_results = collection.query(query_texts=[query], n_results=10, include=["documents", "distances"])
    return query_results

def get_response(context, difficulty, question_type):
    prompt_text = """
Create three {question_type} questions and all the three questions must only be of {difficulty} level based on the given context.


Context: {context}

Questions:
"""
    prompt = PromptTemplate.from_template(prompt_text)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    result = chain.invoke({"context": context,"difficulty":difficulty,"question_type":question_type})
    return result

def get_topics(context):
    prompt_text = """
Transcript:
{context}

Example of a subsection:
4.2: 
Concept

5.1:
Topic

2.3:
Concept

The transcript provided is that of a pdf of a chapter of a textbook identify the name of the chapter and the topics mentioned subsections.
"""
    prompt1 = PromptTemplate.from_template(prompt_text)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    output_parser = StrOutputParser()
    chain = prompt1 | model | output_parser
    result = chain.invoke({"context": context})
    return result

# Streamlit interface
st.title("PDF Question Generator")

chroma_client = chromadb.Client()

st.sidebar.title("Collections in Chroma DB")
# Function to refresh collections
def refresh_collections():
    collections = chroma_client.list_collections()
    collection_names = [col.name for col in collections]
    return collection_names

# Initialize session state for collection names
if "collection_names" not in st.session_state:
    st.session_state.collection_names = refresh_collections()

# Initialize session state for chat messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Button to refresh collections
if st.sidebar.button("Refresh Collections"):
    st.session_state.collection_names = refresh_collections()

# Select collection from sidebar
selected_collection_name = st.sidebar.selectbox("Select your uploaded docs", st.session_state.collection_names)
st.sidebar.warning("If no options are present, upload a doc to select")
# File uploader for PDFs
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Select difficulty level
difficulty_level = st.sidebar.select_slider("Select Difficulty Level", ["easy", "medium", "hard"])

# Select question type
question_type = st.sidebar.selectbox("Select Question Type", ["multiple-choice", "long-essay"])


if st.button("Upload your pdf"):
    if uploaded_files:
        pdf_paths = [file.name for file in uploaded_files]
        try:
            # Initialize Chroma client
            chroma_client = chromadb.Client()

            for pdf in uploaded_files:
                # Create or retrieve a collection named after the PDF file
                original_name = pdf.name.replace(".pdf", "")

# Remove invalid characters and ensure length constraints
                valid_name = re.sub(r'[^a-zA-Z0-9_-]', '_', original_name)

# Ensure it starts and ends with an alphanumeric character
                if not valid_name[0].isalnum():
                   valid_name = 'a' + valid_name
                if not valid_name[-1].isalnum():
                    valid_name = valid_name + 'z'

# Ensure no consecutive periods
                while '..' in valid_name:
                   valid_name = valid_name.replace('..', '.')

# Ensure length constraints
                if len(valid_name) < 3:
                   valid_name = valid_name + '123'
                if len(valid_name) > 63:
                   valid_name = valid_name[:63]

                collection_name = valid_name
                try:
                    collection = chroma_client.get_collection(collection_name)
                    st.write("Collection already exists. Skipping embedding addition and getting the context.")

                except ValueError:
                    # Extract text chunks from all PDFs
                    pdf_chunks = get_all_pdfs_chunks([pdf])
                    st.write("PDF Chunks extracted.")
                    
                    # Add chunks to the collection
                    ids = [f"chunk_{i}" for i in range(len(pdf_chunks))]
                    
                    collection = chroma_client.create_collection(collection_name)
                   
                    collection.add(
                        ids=ids,
                        documents=pdf_chunks
                    )
                    
                    st.write(f"Collection '{collection_name}' created.")
                    st.write(f"Chunks added to Chroma collection: {collection_name}")
            
            
            
        except Exception as e:
            if "rate limit" in str(e).lower():
                st.error("Rate limit exceeded. Please try again later.")
            else:
                st.error(f"An error occurred: {e}")
    else:
       
        
        
        st.warning("Drop at least one pdf to upload")

if st.sidebar.button("Generate Questions"):
    
    if selected_collection_name:
        try:
            # Retrieve the selected collection
            collection = chroma_client.get_collection(selected_collection_name)

            # Retrieve documents from the collection
            x = collection.get()
            y = get_topics(x['documents'])

            # Perform similarity search and generate questions
            search_results = perform_similarity_search(y, collection)
            response = get_response(search_results['documents'], difficulty_level, question_type)
            st.session_state.chat_history.append({"role": "user", "content": "Generate questions based on selected preferences and context"})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        except Exception as e:
            if "rate limit" in str(e).lower():
                st.error("Rate limit exceeded. Please try again later.")
            else:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please select a collection.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
