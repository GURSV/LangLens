from tempfile import NamedTemporaryFile
import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from tools import ImageCaptionTool, ObjectDetectionTool, VisualQuestionAnsweringTool
import base64
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Set background image from a local file
def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    background_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        background-position: center top 0px;
    }}
    </style>
    """
    st.markdown(background_image, unsafe_allow_html=True)

set_background('AI.jpg')

# Initialize agent...
tools = [ImageCaptionTool(), ObjectDetectionTool(), VisualQuestionAnsweringTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0,
    model_name="gpt-3.5-turbo"
)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method='generate'
)

st.title('LangLens ֎')
st.header('Ask a question to an image')
st.subheader("Upload an image")

file = st.file_uploader("", type=["jpeg", "jpg", "png"])

if file:
    st.image(file, use_column_width=True)
    user_question = st.text_input('Ask a question about your image:')

    with NamedTemporaryFile(delete=False, dir='.') as f:
        f.write(file.getbuffer())
        image_path = f.name

        if user_question and user_question != "":
            with st.spinner(text="In progress..."):
                # Formulate input for the VisualQuestionAnsweringTool
                if "detect" in user_question.lower() or "object" in user_question.lower():
                    response = agent.run('Detect objects in this image: {}'.format(image_path))
                elif "describe" in user_question.lower() or "caption" in user_question.lower():
                    response = agent.run('Describe the image: {}'.format(image_path))
                else:
                    # Pass question and image path together for VQA
                    vqa_input = f"{user_question} ### {image_path}"
                    response = agent.run(f"Answer this question about the image: {vqa_input}")
                
                st.write(response)