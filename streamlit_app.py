import streamlit as st
import time
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback

from google import genai
from google.genai import types

st.set_page_config(layout="wide", page_title="LLM Chat App")
st.title("🦜🔗 LLM Chat APP")

endpoint = st.secrets["endpoint"]
deployment_name = "gpt-4o" 
subscription_key = st.secrets["subscription_key"]
api_version = "2024-12-01-preview"

client = genai.Client(api_key=st.secrets["api_key"])

select_model = st.sidebar.selectbox(
    'Choose a LLM model',
    ('GPT 4.0', 'Gemini 2.5 flash')
)

temperature_slider = st.sidebar.slider(
    'Temperature:',
    0.01, 2.00, (0.10)
)
top_p_slider = st.sidebar.slider(
    'Top P:',
    0.01, 1.00, (0.90)
)
max_token_slider = st.sidebar.slider(
    'Max Token:',
    1, 10000, (8000)
)

def generate_response(input_text):
    if(select_model == 'GPT 4.0'):
        model =  AzureChatOpenAI(
            azure_deployment=deployment_name, 
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
            temperature= temperature_slider,
            top_p= top_p_slider,
            max_tokens= max_token_slider
        )
        with get_openai_callback() as cb:
            response = model.invoke(input_text)
            st.session_state.usage["total_tokens"] += cb.total_tokens
            st.session_state.usage["prompt_tokens"] += cb.prompt_tokens
            st.session_state.usage["completion_tokens"] += cb.completion_tokens
            st.session_state.usage["cost"] += cb.total_cost

        return response.content
   

    else :
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{input_text}",
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature= temperature_slider,
                top_p= top_p_slider,
                max_output_tokens= max_token_slider
            ),
        )
        usage = response.usage_metadata
        if usage:
            st.session_state.usage["total_tokens"] += usage.total_token_count
            st.session_state.usage["prompt_tokens"] += usage.prompt_token_count
            st.session_state.usage["completion_tokens"] += usage.candidates_token_count
        return response.text



def chat_stream(prompt):
    response = generate_response(prompt)
    for char in response:
        yield char
        time.sleep(0.02)


def save_feedback(index):
    st.session_state.history[index]["feedback"] = st.session_state[f"feedback_{index}"]


if "usage" not in st.session_state:
    st.session_state.usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}

if "history" not in st.session_state:
    st.session_state.history = []

for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            feedback = message.get("feedback", None)
            st.session_state[f"feedback_{i}"] = feedback
            st.feedback(
                "thumbs",
                key=f"feedback_{i}",
                disabled=feedback is not None,
                on_change=save_feedback,
                args=[i],
            )

if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        response = st.write_stream(chat_stream(prompt))
        st.feedback(
            "thumbs",
            key=f"feedback_{len(st.session_state.history)}",
            on_change=save_feedback,
            args=[len(st.session_state.history)],
        )
    st.session_state.history.append({"role": "assistant", "content": response})

def summarize_history():
    conversation = ""
    for msg in st.session_state.history:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation += f"{role}: {msg['content']}\n"
    summary_prompt = f"Summarize the following conversation briefly:\n\n{conversation}"
    summary = generate_response(summary_prompt)
    return summary


if st.sidebar.button("Summarize"):
    if(st.session_state.history):
        summary = summarize_history()
        st.subheader("Conversation Summary")
        st.write(summary)
    else:
        st.write("No conversation yet to summarize.")   




with st.sidebar:
    st.subheader("💰 Usage")
    st.write(f"Prompt Tokens: {st.session_state.usage['prompt_tokens']}")
    st.write(f"Completion Tokens: {st.session_state.usage['completion_tokens']}")
    st.write(f"Total Tokens: {st.session_state.usage['total_tokens']}")
    st.write(f"Estimated Cost: ${st.session_state.usage['cost']:.6f}")

