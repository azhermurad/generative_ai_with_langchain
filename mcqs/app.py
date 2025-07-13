import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyMuPDFLoader
import tempfile
import json
import base64

# ----------- SETUP -------------
st.set_page_config(page_title="MCQ Auto Checker", layout="centered")

st.title("📄 MCQ Auto-Checker (PDF) using Gemini Pro Vision")
st.markdown("Upload a filled MCQ PDF paper. The system will extract selected answers and compare with the correct answer key.")

# ----------- INPUTS -------------

google_api_key = "AIzaSyAOpKLgJX_J3RSnM5eTWoyU_mJwvQlir1M"
uploaded_file = st.file_uploader("📤 Upload Filled MCQ Paper (PDF)", type="pdf")

# ----------- ANSWER KEY (Manually defined) -------------
answer_key = {
    "Q1": "B", "Q2": "A", "Q3": "C", "Q4": "B"
}

# ----------- VISION PROMPT -------------
prompt = """
You are given an image of an MCQ answer sheet. Your task is to identify and return only the selected options for each question.

Only return the selected choices (like A, B, C, D).

Do not repeat the full questions or all options.

Only include the selected choice for each question (e.g., "Q1: A", "Q2: C").

If no option is clearly selected for a question, write "Qx: Not Answered".
"""

# ----------- PROCESSING -------------
if uploaded_file and google_api_key:
    with st.spinner("🔍 Analyzing PDF..."):

        # Example using a local image file encoded in base64
        image_file_path = "b.png"

        with open(image_file_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        message_local = HumanMessage(
            content=[
                {"type": "image_url", "image_url": f"data:image/png;base64,{encoded_image}"},
            ]
        )
       
       
       
       
        # Initialize Gemini Vision model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            google_api_key=google_api_key
        )

        # Run LLM on the prompt + content
        prompt_template = ChatPromptTemplate([
            ("system", "{prompt}"),
            ("user", "{paper}")
            ])

        chain   = prompt_template | llm 
        response = chain.invoke({"prompt": prompt,"paper": message_local})
        st.write(response)

        # Try parsing output
        try:
            user_answers = json.loads(response.content)
        except Exception as e:
            print(e)
            st.error("❌ Failed to parse LLM output. Raw response:")
            st.text(response.content)
            st.stop()

        # Compare with answer key
        # def evaluate(user_answers, correct_answers):
        #     score = 0
        #     results = {}
        #     for q, correct in correct_answers.items():
        #         user = user_answers.get(q)
        #         correct_bool = (user == correct)
        #         results[q] = {"User": user, "Correct": correct, "is_correct": correct_bool}
        #         if correct_bool:
        #             score += 1
        #     return score, len(correct_answers), results

        # score, total, result = evaluate(user_answers, answer_key)

        # # ----------- RESULTS -------------
        # st.success(f"✅ Score: {score} out of {total}")
        # st.subheader("📊 Detailed Result")

        # for q, data in result.items():
        #     status = "✅" if data["is_correct"] else "❌"
        #     user = data["User"] if data["User"] else "—"
            # st.markdown(f"{status} **{q}** | Your Answer: `{user}` | Correct: `{data['Correct']}`")

else:
    st.info("ℹ️ Please upload a PDF and provide your Google API key to continue.")
