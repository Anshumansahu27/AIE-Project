import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from CssTemplates import css, bot_template, user_template
from langchain.document_loaders import UnstructuredURLLoader

def get_pdf_text(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    return data


def get_text_chunks(data):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(data)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.3)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple URLs",
                       page_icon=":chart_with_upwards_trend:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple URLs :chart_with_upwards_trend:")
    main_placeholder = st.empty()
    user_question = st.text_input("Ask question :")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your URLs")

        urls = []
        for i in range(4):
            url = st.sidebar.text_input(f"URL {i+1}")
            urls.append(url)

        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                main_placeholder.text("Data Loading...Started...✅")
                raw_text = get_pdf_text(urls)

                # get the text chunks
                main_placeholder.text("Text Splitter...Started...✅✅")
                text_chunks = get_text_chunks(raw_text)
                

                # create vector store
                main_placeholder.text("Embedding Vector Started Building...✅✅✅")
                vectorstore = get_vectorstore(text_chunks)
                

                # create conversation chain
                main_placeholder.text("Ready...✅✅✅✅")
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()