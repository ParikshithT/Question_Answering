import streamlit as st
from transformers import pipeline
import PyPDF2

st.title('Question Answering BOT')

def extract_text_from_pdf(pdf_file):
    text = ""
    
    # Open the PDF file in read-binary mode
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    # Loop through each page and extract text
    for page_number in range(len(pdf_reader.pages) ):
        page = pdf_reader.pages[page_number]
        text += page.extract_text()
    
    return text

@st.cache_resource()
def load_qa_model():
    model = pipeline("question-answering")
    return model

qa = load_qa_model()
st.title("Ask Questions about your Text")

upload = st.file_uploader('Upload PDF File', type='pdf')

if upload is not None:
    df = extract_text_from_pdf(upload)

st.write(df)

question = st.text_input("Questions from this article?")
button = st.button("Get me Answers")

with st.spinner("Discovering Answers.."):
    if button and df:
        answers = qa(question=question, context=df)
        st.write(answers['answer'])
