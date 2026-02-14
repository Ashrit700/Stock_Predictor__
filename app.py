#pip install streamlit
import streamlit as st
from predict import predict_intent


st.set_page_config(page_title="ML chatbot")

st.title("Ml powered chatbot")
st.write("Chatbot with intent")
user_input=st.text_input("Enter ")
if " chat history" not in st.session_state:
    st.session_state.chat_history=[]
if st.button("send"):
    if user_input.strip() !="":
        st.session_state.chat_history.append(("You: ",user_input))

        prediction=predict_intent(user_input)

        if prediction=="greet":
            response = "Hello how can i help you!!"
        elif prediction=="weather":
            response="Current weather is ...."
        else:
            response="I didnt understand that"

        st.session_state.chat_history.append(("Bot",response))
for sender, message in st.session_state.chat_history:
    if sender=="You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:**" {message})


