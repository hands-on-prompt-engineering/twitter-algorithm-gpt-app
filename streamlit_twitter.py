
"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import asyncio
import aiohttp
# From here down is all the StreamLit UI.
st.set_page_config(page_title="Twitter Documentation", page_icon=":robot:")
st.header("Twitter Open Source Code Demo")

async def load_chain():
    """Here is the new twitter alogithm https://github.com/twitter/the-algorithm"""
    import os
    import os
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_BASE"] = ""
    os.environ["OPENAI_API_KEY"] = ""
    from langchain.embeddings.openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)
    from langchain.vectorstores import DeepLake
    db = DeepLake(dataset_path="hub://apurvsibal/twitter-algorithm", read_only=True, embedding_function=embeddings)#davitbun
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 20
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import AzureChatOpenAI
    from langchain.schema import HumanMessage
    BASE_URL = ""
    API_KEY = ""
    DEPLOYMENT_NAME = "chat"
    model = AzureChatOpenAI(
        openai_api_base=BASE_URL,
        openai_api_version="2023-03-15-preview",
        deployment_name=DEPLOYMENT_NAME,
        openai_api_key=API_KEY,
        openai_api_type = "azure",
    )
    #model = ChatOpenAI(model='gpt-4') # 'gpt-3.5-turbo',
    qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)
    return qa

#qa = load_chain()
if 'notsomething' in locals():
    something = 1
else:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    qa = loop.run_until_complete(load_chain())
    notsomething = 1


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def get_text():
    input_text = st.text_input("You: ", "What does favCountParams do?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = qa({"question": user_input, "chat_history": []})
    #st.session_state["chat_history"].append((user_input, output['answer']))
    #output = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output['answer'])

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")