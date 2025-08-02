from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def main():
    st.set_page_config(page_title="Document Analyzer")

    # Custom CSS for colors and styling
    st.markdown(
        """
        <style>
        body {
            background-color: #FAFAFA;
            font-family: Arial, sans-serif;
        }
        h1 {
            color: #335C47;
            margin-bottom: 0;
        }
        .subtitle {
            font-size: 14px;
            color: #0E2B3F;
            margin-top: 2px;
            margin-bottom: 24px;
        }
        h2 {
            color: #0E2B3F !important;
            border-bottom: 2px solid #0E2B3F;
            padding-bottom: 6px;
            margin-top: 40px;
        }
        div.stTextInput > label > div {
            color: #0E2B3F;
            font-weight: 600;
        }
        div.stTextInput > div > input {
            border: 1.5px solid #335C47;
            border-radius: 6px;
            padding: 8px;
        }
        div.stFileUploader > label {
            font-weight: 600;
            color: #0E2B3F;
        }
        button, div.stButton > button {
            background-color: #335C47 !important;
            color: white !important;
            border-radius: 6px !important;
            padding: 10px 18px !important;
            font-weight: 600 !important;
            border: none !important;
            cursor: pointer;
        }
        a {
            color: #0E2B3F !important;
            text-decoration: none !important;
        }
        a:hover {
            text-decoration: underline !important;
        }
        .footer {
            font-size: 12px;
            color: #666666;
            text-align: center;
            padding-top: 20px;
            border-top: 1px solid #ccc;
            margin-top: 40px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Document Analyzer")
    st.markdown(
        "<div class='subtitle'>Upload your PDF, then ask any question about its contents — powered by AI to get quick, accurate insights.</div>",
        unsafe_allow_html=True
    )

    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        if not chunks:
            st.write("No text chunks were extracted from the PDF.")
            return

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.markdown("<h2>Ask me anything about your document!</h2>", unsafe_allow_html=True)

        user_question = st.text_input("Enter your question here:")

        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback():
                response = chain.run(input_documents=docs, question=user_question)
                st.write(response)
    else:
        st.write("Please upload a PDF to get started.")

    st.markdown(
        """
        <div class="footer">
            Brought to you by Low Country Cyber Solutions 🛡️ Savannah, GA
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
