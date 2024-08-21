import uuid

import snowflake.connector
import streamlit as st
from dotenv import dotenv_values
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_snowpoc.chat_message_histories import SnowflakeChatMessageHistory
from langchain_snowpoc.document_loaders.snowflake_stage_directory import (
    SnowflakeStageDirectoryLoader,
)
from langchain_snowpoc.embedding import SnowflakeEmbeddings
from langchain_snowpoc.llms import SQLCortex
from langchain_snowpoc.vectorstores import SnowflakeVectorStore

config = dotenv_values(".env")

MODEL_EMBEDDINGS = config["MODEL_EMBEDDINGS"]
VECTOR_LENGTH = config["VECTOR_LENGTH"]
STAGE_NAME = config["STAGE_NAME"]
CHAT_HISTORY_TABLE = config["CHAT_HISTORY_TABLE"]
CONNECTION_NAME = config["CONNECTION_NAME"]
MODEL_NAME = config["MODEL_NAME"]


GITHUB_ROOT = "https://github.com/Snowflake-Labs/sfquickstarts"
GITHUB_ASSETS = (
    "ask-questions-to-your-documents-using-rag-with-snowflake-cortex/assets/"
)
GITHUB_URL = f"{GITHUB_ROOT}/blob/9b1369fe6198ee62e0fc8d879209bc4c0b5b9cb6/site/sfguides/src/{GITHUB_ASSETS}"
PDF_LIST = [
    f"{GITHUB_URL}/Mondracer_Infant_Bike.pdf",
    f"{GITHUB_URL}/Premium_Bicycle_User_Guide.pdf",
    f"{GITHUB_URL}/Ski_Boots_TDBootz_Special.pdf",
    f"{GITHUB_URL}/The_Ultimate_Downhill_Bike.pdf",
    f"{GITHUB_URL}/The_Xtreme_Road_Bike_105_SL.pdf",
]

st.set_page_config(layout="wide")

st.title("Document Chat Bot")


@st.cache_resource
def get_connection(connection_name):
    return snowflake.connector.connect(
        connection_name=connection_name,
    )


def download_sample_pdfs():
    """Setup stage with pdf files. The stage must exist"""
    import os

    import requests

    with st.spinner("Settin' files in stage..."):

        tempdir = os.path.join(os.getcwd(), "temp_pdfs")
        os.makedirs(tempdir, exist_ok=True)

        with get_connection(CONNECTION_NAME).cursor() as cs:
            cs.execute(f"DROP STAGE IF EXISTS {STAGE_NAME}")
            cs.execute(f"CREATE STAGE IF NOT EXISTS {STAGE_NAME}")

            for url in PDF_LIST:
                filename = os.path.basename(url)
                local_file_name_with_path = f"{tempdir}/{filename}"

                if not os.path.exists(local_file_name_with_path):
                    r = requests.get(f"{url}?raw=true", stream=True)

                    with open(local_file_name_with_path, "wb") as fd:
                        for chunk in r.iter_content(1024):
                            fd.write(chunk)

                cs.execute(
                    f"PUT file://{local_file_name_with_path} @{STAGE_NAME}/descriptions/bikes/ auto_compress=false"
                )
                print(cs.fetchall())


@st.cache_resource
def get_document_splits(_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400,
        length_function=len,
        is_separator_regex=False,
    )

    documents = text_splitter.split_documents(_docs)
    return documents


@st.cache_resource
def get_docs_from_stage():
    with st.spinner("Gettin' documents from stage..."):
        ssdl = SnowflakeStageDirectoryLoader(
            stage_directory=f"@{STAGE_NAME}/descriptions/bikes/",
            connection=get_connection(CONNECTION_NAME),
        )
        docs = ssdl.load()
        return docs


def get_history_aware_retriever(_llm, _retriever):
    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        _llm, _retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def get_question_answer_chain(_llm):
    ### Answer question ###
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(_llm, qa_prompt)
    return question_answer_chain


### Statefully manage chat history ###
def get_session_history_wrap(
    snowflake_connection, chat_history_table: str = "chat_history"
):
    def _get_session_history(
        session_id: str,
    ) -> BaseChatMessageHistory:
        return SnowflakeChatMessageHistory(
            chat_history_table,
            session_id,
            connection=snowflake_connection,
        )

    return _get_session_history


get_session_history = get_session_history_wrap(
    get_connection(CONNECTION_NAME), CHAT_HISTORY_TABLE
)


@st.cache_resource
def do_ask():
    llm = SQLCortex(
        connection=get_connection(CONNECTION_NAME),
        model=MODEL_NAME,
    )

    embeddings = SnowflakeEmbeddings(
        connection=get_connection(CONNECTION_NAME),
        model=MODEL_EMBEDDINGS,
    )

    documents = get_document_splits(get_docs_from_stage())

    retriever = SnowflakeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        vector_length=VECTOR_LENGTH,
        connection=get_connection(CONNECTION_NAME),
    ).as_retriever()

    rag_chain = create_retrieval_chain(
        get_history_aware_retriever(llm, retriever), get_question_answer_chain(llm)
    )

    SnowflakeChatMessageHistory.create_tables(
        get_connection(CONNECTION_NAME), CHAT_HISTORY_TABLE
    )

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return lambda question, session_id: (
        conversational_rag_chain.invoke(
            {"input": question},
            config={
                "configurable": {
                    "session_id": session_id,
                }
            },
        )["answer"]
        if question != "TECHNICAL_QUESTION"
        else lambda a, b: f"{a}, {b}"
    )


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


if "setup_done" not in st.session_state:
    download_sample_pdfs()
    do_ask()("TECHNICAL_QUESTION", session_id=st.session_state.session_id)
    st.session_state.setup_done = True

with st.sidebar:
    st.write(f"Using model: {MODEL_NAME}")
    st.write(f"Session id: {st.session_state.session_id}")


prompt = st.chat_input()

if prompt:
    do_ask()(prompt, session_id=st.session_state.session_id)

    for msg in get_session_history(st.session_state.session_id).messages:
        agent_type = msg.__class__.__name__
        if agent_type == "HumanMessage":
            st.chat_message("user").write(msg.content)
        if agent_type == "AIMessage":
            st.chat_message("assistant").write(msg.content)
