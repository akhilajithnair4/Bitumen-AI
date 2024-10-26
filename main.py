import os

# import pickle
import time
# from langchain import OpenAI
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores.faiss import FAISS
from  langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain

import dill
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

# Try to get the API key from st.secrets (for deployment) or .env (for local development)
api_key = os.getenv("OPENAI_API_KEY")

# If no valid API key, ask the user to enter it
if not api_key:
    api_key = st.text_input("Please enter your OpenAI API key:", type="password")
    if not api_key:
        st.error("No API key provided. Please provide an API key to continue.")
        st.stop()

# Set API key in environment if necessary (for libraries that use os.environ)
os.environ['OPENAI_API_KEY'] = api_key



llm=OpenAI(temperature=0.7,max_tokens=500)

url=["https://www.investing.com/charts/forex-charts","https://atdmco.com/penetration/bitumen-price/","https://www.petronaftco.com/bitumen-price/","https://www.investing.com/news/commodities-news"]

loader=UnstructuredURLLoader(urls=url)
data=loader.load()

# split the data into chunks

text_splitter=RecursiveCharacterTextSplitter(
    separators=['\n\n','\n','-',','],chunk_size=1000
)

docs=text_splitter.split_documents(data)

# Create Embedings and Save it in Vector Database

embedings= OpenAIEmbeddings()


vector_index=FAISS.from_documents(docs,embedings)

# vector_store_embeddings.lock = None

# # Save the FAISS index to a pickle file
# file_path="vector_index.pkl"
# with open(file_path,"wb") as f:
#     dill.dump(vector_store_embeddings, f)

# Save the FAISS index to a file

# Save the documents and FAISS index separately
# Save the documents and FAISS index separately
file_path_docs = "docs.pkl"
faiss_index_path = "faiss_index"

# Save documents
with open(file_path_docs, "wb") as f_docs:
    dill.dump(docs, f_docs)

# Save FAISS index separately
vector_index.save_local(faiss_index_path)


# Header with email and phone
st.markdown("""
<div class="header">
    <div>Bitumen Impex</div>
    <div>Email: sellingbitumen@gmail.com | Phone: +91 9349037606</div>
</div>
""", unsafe_allow_html=True)
# Custom CSS styling for a refined answer display
st.markdown("""
    <style>
        /* Answer styling */
        .answer {
            background-color: #F1F1F1;
            padding: 15px;
            border-radius: 8px;
            font-size: 16px;  /* Adjusted to a smaller, refined font size */
            color: #34495E;
            margin-top: 15px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Bitumen Impex -Commodity AI")
st.write("")  # Adds one empty line of space
st.write("")  # Adds another empty line of space
st.write(" ")  # Adds another empty line of space
st.write("")  # Adds another empty line of space
st.write("")  # Adds another empty line of space
st.write("")  # Adds another empty line of space

main_placeholder=st.empty()

process_url_clicked=st.button("Ask Commodity AI")

query=main_placeholder.text_input("Question: ")

if query:
    # if os.path.exists(file_path):
    if os.path.exists(file_path_docs) and os.path.exists(faiss_index_path):
        with open(file_path_docs, 'rb') as f_docs:
            docs = dill.load(f_docs)  # Load documents

            # Reload FAISS index and recreate embeddings
            vector_index = FAISS.load_local(faiss_index_path, embedings,allow_dangerous_deserialization=True)
           

            chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vector_index.as_retriever())

            result =chain.invoke({"question":query},return_only_outputs=True)

            st.header("Answer")
            # Display the answer with refined styling
            st.markdown(f"<div class='answer'>{result['answer']}</div>", unsafe_allow_html=True)





