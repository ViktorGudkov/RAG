from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import streamlit as st
from dotenv import load_dotenv
import base64
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.gigachat import GigaChat
from langchain.chains import create_retrieval_chain
from docx import Document as DocxDocument
from langchain.schema import ChatMessage
import logging
from langchain_community.embeddings.gigachat import GigaChatEmbeddings

load_dotenv()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

client_id = os.environ['CLIENT_ID_CM']
secret = os.environ['CLIENT_SECRET_CM']
password = os.environ['password']
credentials = f"{client_id}:{secret}"
auth= base64.b64encode(credentials.encode('utf-8')).decode('utf-8')

prompt = ChatPromptTemplate.from_template('''Ответь на вопрос пользователя. \
Используй при этом только информацию из контекста. Если в контексте нет \
информации для ответа, сообщи об этом пользователю.
Контекст: {context}
Вопрос: {input}
Ответ:'''
)

model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
# embedding = HuggingFaceEmbeddings(model_name=model_name,
#                                   model_kwargs=model_kwargs,
#                                   encode_kwargs=encode_kwargs)


embedding=GigaChatEmbeddings(
        credentials=auth,
        verify_ssl_certs=False,
        scope='GIGACHAT_API_CORP',
        profanity_check=False
    )


vector_store=FAISS.load_local('vector_store',embedding, allow_dangerous_deserialization=True)

embedding_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

st.title("Gigachat бот")

# Проверка пароля пользователя
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    entered_password = st.text_input("Введите пароль для доступа:", type="password")
    if entered_password == password:
        st.session_state.authenticated = True
        st.success("Пароль верный! Добро пожаловать!")
    elif entered_password:
        st.error("Неверный пароль. Попробуйте снова.")
        st.stop()

# инициалиация истории чата
if "messages" not in st.session_state:
    st.session_state.messages = [
        ChatMessage(role="assistant", content="Напечатайте Ваш вопрос"),
    ]

# отображение сообщений чата из истории при повторном запуске приложения
for message in st.session_state.messages:
    with st.chat_message(message.role):
        st.markdown(message.content)

if input := st.chat_input():
    chat = GigaChat(credentials=auth,
                model= 'GigaChat-Pro',
                verify_ssl_certs=False,
                scope='GIGACHAT_API_CORP',
                profanity_check=False
                )
    document_chain = create_stuff_documents_chain(
    llm=chat,
    prompt=prompt
    )

    message = ChatMessage(role="user", content=input)
    st.session_state.messages.append(message)

    with st.chat_message(message.role):
        st.markdown(message.content)
    with st.spinner("Бот обрабатывает ваш запрос..."):
        qa_chain = create_retrieval_chain(embedding_retriever, document_chain)
        response = qa_chain.invoke({'input': input})
    otvet = response['answer']

    message = ChatMessage(role="assistant", content=otvet)
    st.session_state.messages.append(message)

    with st.chat_message(message.role):
        message_placeholder = st.empty()
        message_placeholder.markdown(message.content)

    st.session_state.token = chat._client.token
    chat._client.close()
