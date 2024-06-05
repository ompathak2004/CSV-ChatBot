import os
import pickle
import streamlit as st
import tempfile
import pandas as pd
import asyncio

from streamlit_chat import message
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS

st.set_page_config(layout="wide", page_icon=":page_with_curl:", page_title="CSV-ChatBot")


st.markdown(
    """
    <style>
    .centered-header {
        text-align: center;
        font-size: 2.5em;
        color: ivory;
        margin-top: 50px;
    }
    </style>
    <h1 class='centered-header'>CSV-ChatBot, Talk to your CSV-data</h1>
    """,
    unsafe_allow_html=True
)
user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key ",
    placeholder="Paste your openAI API key",
    type="password")

def main():
    
    if user_api_key == "":
        
        st.markdown(
            "<div style='text-align: center;'><h4>Enter your OpenAI API key to start</h4></div>",
            unsafe_allow_html=True)
        
    else:
        os.environ["OPENAI_API_KEY"] = user_api_key
        
        uploaded_file = st.sidebar.file_uploader("upload", type="csv", label_visibility="hidden")
        
        if uploaded_file is not None:
            def show_user_file(uploaded_file):
                file_container = st.expander("Your CSV file :")
                shows = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)
                file_container.write(shows)
                
            show_user_file(uploaded_file)
            
        else :
            st.sidebar.info(
            "Upload your CSV file to get started"
            )
    
        if uploaded_file :
            try :
                def storeDocEmbeds(file, filename):
                    
                    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
                        tmp_file.write(file)
                        tmp_file_path = tmp_file.name

                    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
                    data = loader.load()

                    embeddings = OpenAIEmbeddings()
                    
                    vectors = FAISS.from_documents(data, embeddings)
                    os.remove(tmp_file_path)

                    with open(filename + ".pkl", "wb") as f:
                        pickle.dump(vectors, f)
                    
                def getDocEmbeds(file, filename):
                    
                    if not os.path.isfile(filename + ".pkl"):
                        storeDocEmbeds(file, filename)
                    
                    with open(filename + ".pkl", "rb") as f:
                        global vectores
                        vectors = pickle.load(f)
                        
                    return vectors

                def conversational_chat(query):
                    
                    result = chain({"question": query, "chat_history": st.session_state['history']})
                    
                    st.session_state['history'].append((query, result["answer"]))
                    
                    print("Log: ")
                    print(st.session_state['history'])
                    
                    return result["answer"]

                with st.sidebar.expander("🛠️ Settings", expanded=False):
                    
                    if st.button("Reset Chat"):
                        st.session_state['reset_chat'] = True

                    MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','gpt-4'])

                if 'history' not in st.session_state:
                    st.session_state['history'] = []

                if 'ready' not in st.session_state:
                    st.session_state['ready'] = False
                    
                if 'reset_chat' not in st.session_state:
                    st.session_state['reset_chat'] = False
                
                if uploaded_file is not None:

                    with st.spinner("Processing..."):

                        uploaded_file.seek(0)
                        file = uploaded_file.read()
                        
                        vectors = getDocEmbeds(file, uploaded_file.name)

                        chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name=MODEL, streaming=True),
                                                                      retriever=vectors.as_retriever(), chain_type="stuff")

                    st.session_state['ready'] = True

                if st.session_state['ready']:

                    if 'generated' not in st.session_state:
                        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

                    if 'past' not in st.session_state:
                        st.session_state['past'] = ["Hey ! 👋"]

                    response_container = st.container()
                    
                    container = st.container()

                    with container:
                        
                        with st.form(key='my_form', clear_on_submit=True):
                            
                            user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
                            submit_button = st.form_submit_button(label='Send')
                            
                            if st.session_state['reset_chat']:
                                
                                st.session_state['history'] = []
                                st.session_state['past'] = ["Hey ! 👋"]
                                st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]
                                response_container.empty()
                                st.session_state['reset_chat'] = False

                        if submit_button and user_input:
                            
                            output = conversational_chat(user_input)
                            
                            st.session_state['past'].append(user_input)
                            st.session_state['generated'].append(output)

                    if st.session_state['generated']:
                        
                        with response_container:
                            
                            for i in range(len(st.session_state['generated'])):
                                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()