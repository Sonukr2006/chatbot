import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os



# Set up the Streamlit app
st.set_page_config(page_title="LangChain Chatbot with Groq")

# Title
st.title("LangChain Chatbot with Groq")
st.markdown(
    """
    This is a simple chatbot application built using LangChain and Groq. 
    Enter your Groq API key, select a model, and start chatting!
    """
)

with st.sidebar:
    st.header("Settings")

    # API Key input
    api_key = st.text_input("Enter your Groq API Key", type="password", help="You can find your API key from console.groq.com")

    if not api_key:
        st.warning("Please enter your Groq API Key.")

    # Model selection
    model_name = st.selectbox("Select a model", ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "meta-llama/llama-prompt-guard-2-22m"], index=0, help="Choose the model you want to use for the chatbot.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    
if "messages" not in st.session_state:
    st.session_state.messages = []


#initialize the chat model
@st.cache_resource
def get_chat_model(api_key, model_name):
    if not api_key:
        return None
    
    llm =ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        streaming=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant powered by Groq. Answer the user's questions to the best of your ability. Answer in a concise and clear manner. If you don't know the answer, say 'I don't know'."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()
    return chain

# get chain

chain = get_chat_model(api_key, model_name)


if not chain:
    st.warning("Please enter your Groq API Key to initialize the chatbot.")
    st.markdown("You can find your API key from [console.groq.com](https://console.groq.com).")
else:
    ## Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    ## User input
    if prompt := st.chat_input("Ask me anything!"):
        # 1. User message display karein
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Existing messages se chat_history list taiyar karein (current prompt se pehle ke messages)
        chat_history = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))

        # 3. Current user prompt ko session state me add karein
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 4. Stream response (dono keys pass karni hain)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in chain.stream({
                    "input": prompt,
                    "chat_history": chat_history   # <-- Yeh pass hona zaroori hai
                }):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "|")  # Streaming indicator

                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
