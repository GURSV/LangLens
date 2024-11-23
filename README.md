# LangLens ֎

LangLens is an AI-powered model combining OpenAI's GPT, Salesforce's Visual Question Answering (VQA) base model, Fine Tuned Salesforce's VQA model & Facebook's detr-resnet-50. It offers the ability to:
- Generate image captions.
- Detect objects in images with high accuracy.
- Answer basic image-related questions.

## Features
- **Image Captioning:** Provides detailed captions for uploaded images.
- **Object Detection:** Identifies objects within images effectively.
- **Question Answering:** Responds to queries about the content of an image.

## Installation

1. Clone the repository:
   - git clone https://github.com/GURSV/LangLens.git
   - cd LangLens

2. Install dependencies:
   - pip install -r requirements.txt

## Usage
Run the fine-tuning script:
- python fine_tune_colab.py

Run the main script:
- python main.py

This will enable image processing and interactive question answering.

Additional utilities are provided in:
- tools.py - Supplementary tools for model interaction.

Dataset:
- The model uses a CSV dataset for fine-tuning. Ensure the dataset is formatted appropriately for training.

Images folder:
- images/ - For training the fine-tune model
- images-for-test/ - For testing the project

Do - streamlit run main.py for running the project locally (http://localhost:8501)

View and working of the application

![image](https://github.com/user-attachments/assets/4867d31b-9852-4495-b70f-9588b82675cd)

![image](https://github.com/user-attachments/assets/eb4a22d1-466c-4edd-9eaf-b5b2d6f0a540)

![image](https://github.com/user-attachments/assets/3812bb5c-aea3-4fca-b4e7-313fbaa8c20a)

![image](https://github.com/user-attachments/assets/40a1c150-2ab3-4f47-9515-b78129848050)

![image](https://github.com/user-attachments/assets/36d1ca7a-5956-46bb-8d3c-588deeed0c49)

etc...

Thank you.
