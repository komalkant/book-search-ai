# Install all libraries by running in the terminal: pip install -q -r ./requirements.txt
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma

from langchain.vectorstores import Chroma
import chromadb


# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks, file_name):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
    st.write(f'file_name: ${file_name}')
    # vector_store = Chroma.from_documents(chunks, embeddings)
    # vector_store = Chroma.from_text(chunks, embeddings)
    # if you want to use a specific directory for chromadb
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./'+file_name)
    return vector_store


def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    print(retriever)
    answer = chain.invoke(q)
    return answer['result']


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # check prices here: https://openai.com/pricing
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00002:.6f}')
    return total_tokens, total_tokens / 1000 * 0.00002


# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.image('data:image/jpeg;base64,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')
    st.subheader('Question-Answering Application 🤖')
    st.markdown("""
        <style>
            .reportview-container {
                margin-top: -2em;
            }
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        # api_key = st.text_input('OpenAI API Key:', type='password')
        # if api_key:
        #     os.environ['OPENAI_API_KEY'] = api_key

        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size number widget
        # chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        chunk_size = 512

        # k number input widget
        # k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        k = 3

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./files/', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                source = file_name
                chunks = chunk_data(data, chunk_size=chunk_size)
                splitFile_name = file_name.split('/')
                getFile_name = splitFile_name[2].split('.')
                st.write(f'source: ${getFile_name[0]}')
                # st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')
                # st.write(f'source: ${getFile_name[0]}')
                tokens, embedding_cost = calculate_embedding_cost(chunks)
                # st.write(f'chunks: ${chunks}')

                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embeddings(chunks, getFile_name[0])
                vector_store.persist()
                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

    # user's question text input widget
    # q = st.text_input('Ask a question about the content of your file:')
    # # if q: # if the user entered a question and hit enter
    if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
        vector_store = st.session_state.vs
    #     # st.write(f'k: {k}')
    #     answer = ask_and_get_answer(vector_store, q, k)

    #     # text area widget for the LLM answer
    #     # st.text_area('Answer: ', value=answer)

    #     # st.divider()

    #     # if there's no chat history in the session state, create it
    #     if 'history' not in st.session_state:
    #         st.session_state.history = ''

    #     # the current question and answer
    #     # the current question and answer
    #     value = f'Q: {q} \nA: {answer}'

    #     st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
    #     h = st.session_state.history

    #     # text area widget for the chat history
    #     st.text_area(label='Chat History', value=h, key='history', height=400)

# run the app: streamlit run ./chat_with_documents.py

# st.title("Echo Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the content of your file:"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # st.write(f'vector_store: ${vector_store}')
    # chroma_db = chromadb.PersistentClient(path='./SBI_FASTag_Full_KYC_Application_Form')
    # collection = chroma_db.get_collection(name="langchain")
    # st.write(f'coll: ${collection}')
    # Get the metadata list
    # metadata_list = collection.get()['metadatas']
    # print(metadata_list)
    answer = ask_and_get_answer(vector_store, prompt, k)
    response = f" {answer}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})