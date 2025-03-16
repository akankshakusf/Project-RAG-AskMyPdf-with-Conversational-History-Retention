#import all the packages 

import os 
import streamlit as st
import certifi  # Import certifi

# Set SSL certificate file path using certifi
os.environ["SSL_CERT_FILE"] = certifi.where()



#---------------------------------------------------
#for steamlit deployment only 
HF_TOKEN = st.secrets["HF_TOKEN"] #Comment both key when runing on vs code 
groq_api_key = st.secrets["GROQ_API_KEY"] #Comment both key when runing on vs code 
#---------------------------------------------------

####  RAG libraries START ####
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # Use FAISS for vector store
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

#Importing chat history module to integrate past messages into a LangChain pipeline 

# Community-contributed chat history storage
from langchain_community.chat_message_histories import ChatMessageHistory

# Base class for custom chat history implementations  
from langchain_core.chat_history import BaseChatMessageHistory

# Enables chat memory in LangChain pipelines   
from langchain_core.runnables.history import RunnableWithMessageHistory  

# Placeholder for inserting chat history into a prompt template
from langchain_core.prompts import MessagesPlaceholder 

####  RAG libraries END  ####

### load environment variables
from dotenv import load_dotenv
load_dotenv()

#get all the keys needed
huggingface_api_key=os.getenv("HF_TOKEN")
groq_api_key=os.getenv("GROQ_API_KEY")

#initialize hf embedding model 
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


#---------------#  Setting up the Streamlit app  #---------------#

st.title("AskMyPDF RAG with Chat History") #displayed on app
st.write("Upload pdf's and chat with their content") #displayed on app

# input the Groq api key 
api_key=st.text_input("Enter the Groq API key", type="password",value=groq_api_key) #display on app

#check if api is inserted:
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-it")

    ## api key is present so session starts: CHAT INTERFACE ##

    #since session has started set a default session_id
    session_id=st.text_input("Session ID", value="default_session")

    #statefully managing chat history 
    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_files=st.file_uploader("Choose a pdf file",type="pdf",accept_multiple_files=True)

    #Process uploaded pdf
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,'wb') as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            #Data Ingestion
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

        #Data Transformation into chunks
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits=text_splitter.split_documents(documents)

        #Vector Embeddings
        # Vector Embeddings and FAISS Vector Store
        vectorstore = FAISS.from_documents(documents=splits,embedding=embeddings)
        retriever=vectorstore.as_retriever()


        ### Design the Prompts for LLM
        contextualize_q_system_prompt=(
                "Given a chat history and the latest user question"
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt=ChatPromptTemplate.from_messages(
                [
                    ("system",contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human","{input}")
                ]
            )
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        ## Question Answer Prompt

        system_prompt=(
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "/n/n"
            "{context}"
        )

        # qa_prompt is used to generate the final answer based on the retrieved information and chat history.

        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]

        )

        # * Here we are now going to make changes we will pass qa_prompt that is history aware
        # * we will also pass history_aware_retriever as it keeps account of past conversations

        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversation_rag_chain=RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"

        )

        user_input=st.text_input("Your Question:")
        if user_input:
            session_history=get_session_history(session_id)
            response = conversation_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
        )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)

else:
    st.warning("Please enter the Groq API Key")











