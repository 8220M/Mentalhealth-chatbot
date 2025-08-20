import streamlit as st

st.title('Mental Health Emotion Detection Chatbot')
txt = st.text_area('Say something…')
if st.button('Analyze') and txt.strip():
    st.success('Predicted emotion: neutral')  # placeholder
st.caption('LSTM classifier + optional LLM routing (LangChain/OpenAI).')
