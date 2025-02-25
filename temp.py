import os
import re
import cv2
import pickle
import base64
import numpy as np
import easyocr
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from groq import Groq
from fastai.vision.all import load_learner, PILImage
from pathlib import Path
import asyncio
import dill  # Use dill as the pickle module

# -------------------------------------------
# Patch pathlib.PosixPath to WindowsPath on Windows
# -------------------------------------------
if os.name == 'nt':
    import pathlib
    class PatchedPosixPath(pathlib.WindowsPath, pathlib.PurePosixPath):
        pass
    pathlib.PosixPath = PatchedPosixPath
    Path = pathlib.Path

# -------------------------------------------
# Define any custom functions required by the exported learner
# -------------------------------------------
def get_x(r):
    # Dummy definition for get_x; replace with your actual function if needed.
    return r

def get_y(r):
    # Dummy definition for get_y; ensure it matches what was used during training.
    return r['Finding Labels'].split('|')

# -------------------------------------------
# 0) Diagnomate Header Section (Home Page)
# -------------------------------------------
@st.cache_data
def show_banner():
    try:
        with open("assets/med.jpeg", "rb") as file_:  # Update path if needed
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
        return data_url
    except Exception as e:
        st.error(f"Banner not found: {e}")
        return None

# -------------------------------------------
# 1) Environment Setup & Global Config
# -------------------------------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in your .env file or environment variables.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
reader = easyocr.Reader(['en'], gpu=False)

MODEL_PATH = 'trained_model.pkl'
try:
    learn = load_learner(MODEL_PATH, cpu=True, pickle_module=dill)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

MEDICINE_INVENTORY = {
    "Paracetamol": {"stock": 50, "dosage": "500mg", "price": 20},
    "Ibuprofen":   {"stock": 25, "dosage": "400mg", "price": 30},
    "Aspirin": {"stock": 10, "dosage": "500mg", "price": 100},
}

INITIAL_RESPONSE = "Hi! 🤖 I'm your Health Assistant. Tell me your symptoms, and I'll help predict possible diseases."
CHAT_CONTEXT = (
    "You are a health assistant chatbot trained to predict potential diseases based on user symptoms. "
    "Always advise users to consult a doctor for accurate diagnosis. Avoid giving prescriptions."
)

st.set_page_config(page_title="Diagnomate", page_icon="🩺", initial_sidebar_state='expanded')

# -------------------------------------------
# 2) Sidebar Navigation
# -------------------------------------------
st.sidebar.title("Diagnomate")
option = st.sidebar.radio("Select a page:", 
                            ["Home", "Pharmacist's Assistant", "Diagnostic Assistant", "X-ray Analyser"])

# -------------------------------------------
# Home Section (Diagnomate Banner and Description)
# -------------------------------------------
# Home Section (Diagnomate Banner and Description)
if option == "Home":
    st.title("💡 Diagnomate: Your AI Health Diagnostic Companion")
    st.subheader("💖 Empowering Proactive Healthcare Decisions")
    st.markdown("Diagnomate leverages advanced 🤖 AI and deep learning to help you interpret 🧾 medical images and diagnose symptoms accurately. Explore our features below! 🌟")
    st.divider()
    
    banner_data = show_banner()
    if banner_data:
        st.markdown(
            f'<img src="data:image/jpeg;base64,{banner_data}" alt="Diagnomate Banner">',
            unsafe_allow_html=True,
        )
    st.divider()
    
    st.subheader("⚠️ Problems in Healthcare")
    st.markdown("**💊 Challenges in Prescription Interpretation and Pharmacy Operations:** The healthcare industry often faces challenges related to the accurate interpretation of handwritten prescriptions, leading to potential errors in medication dispensing. Additionally, the manual process of matching prescriptions with available inventory is time-consuming and prone to human errors, causing delays in order fulfillment and inefficiencies in pharmacy operations.")
    st.markdown("**🧪 Complexities in Medical Diagnostics:** There is a risk of human error in analyzing medical images, which can lead to misdiagnosis or overlooked conditions. Limited access to experienced specialists, especially in remote or underserved areas, further exacerbates the challenge of timely and accurate diagnosis.")
    st.divider()
    
    st.subheader("💡 Solution")
    st.markdown("**💊 Pharmacist Assistant**")
    st.markdown("Pharmacist’s Assistant tool streamlines the process of extracting medicine information from prescription images and assists pharmacists in quickly matching prescribed medicines with existing inventory, ultimately simplifying order creation.")
    st.markdown("**🧠 AI Techniques and Algorithms Used:**")
    st.markdown(" OCR (Optical Character Recognition) | Image Preprocessing (OpenCV) | CLAHE (Contrast Limited Adaptive Histogram Equalization)")
    st.subheader("📊 Result")

    # Display images in each column using Pathlib to build the paths
    col1, col2 = st.columns(2)
    with col1:
        image_path1 = Path("assets") / "Screenshot 2025-02-25 193504.png"
        st.image(str(image_path1), caption="📋 Result Image 1", use_container_width=True)
    with col2:
        image_path2 = Path("assets") / "Screenshot 2025-02-25 195746.png"
        st.image(str(image_path2), caption="📋 Result Image 2", use_container_width=True)
    
    st.markdown("**🩺 Diagnosis Assistant**")
    st.markdown("The Diagnosis Assistant addresses the challenge of self-diagnosis and preliminary health consultations. Often, individuals struggle to describe symptoms accurately or may not have immediate access to medical professionals. Additionally, it supports healthcare professionals by acting as a decision-support tool, helping them quickly analyze patient-described symptoms, streamline initial assessments, and reduce diagnostic errors.")
    st.markdown("**🧠 AI Techniques and Algorithms Used:**")
    st.markdown(" Large Language Model (LLM) – llama3-8b-8192 | Prompt Engineering")
    st.subheader("📊 Result")
    
    col1, col2 = st.columns(2)
    with col1:
        image_path3 = Path("assets") / "Screenshot 2025-02-25 201228.png"
        st.image(str(image_path3), caption="📋 Result Image 1", use_container_width=True)
    with col2:
        image_path4 = Path("assets") / "Screenshot 2025-02-25 165841.png"
        st.image(str(image_path4), caption="📋 Result Image 2", use_container_width=True)
    
    st.markdown("**🖼️ X-ray Analyzer**")
    st.markdown("The X-ray Analyser addresses challenges in medical diagnostics by reducing human error, speeding up the analysis process, and ensuring consistent evaluations of X-ray images. It aids healthcare professionals in early detection of conditions, even in regions lacking specialized radiologists, enabling faster and more accurate diagnoses.")
    st.markdown("**🧠 AI Techniques and Algorithms Used:**")
    st.markdown("Convolutional Neural Networks (CNNs)")
    st.subheader("📊 Result")
    
    col1, col2 = st.columns(2)
    with col1:
        image_path5 = Path("assets") / "Screenshot 2025-02-25 203846.png"
        st.image(str(image_path5), caption="📋 Result Image 1", use_container_width=True)
    with col2:
        image_path6 = Path("assets") / "Screenshot 2025-02-25 175402.png"
        st.image(str(image_path6), caption="📋 Result Image 2", use_container_width=True)


# 3) Pharmacist’s Assistant
# -------------------------------------------
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img_array = np.array(pil_image)
    if len(img_array.shape) == 2:
        gray = img_array
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY) if img_array.shape[2] == 4 \
               else cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 31, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened

def pharmacist_assistant():
    st.title("Pharmacist’s Assistant: Prescription OCR & Order Creation")
    uploaded_file = st.file_uploader("Upload a prescription image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        pil_image = Image.open(uploaded_file)
        # Display a smaller version of the uploaded image
        st.image(pil_image, caption="Uploaded Image", width=200)
        # No longer displaying preprocessed image
        ocr_result = reader.readtext(np.array(pil_image), detail=0)
        extracted_text = " ".join(ocr_result)
        st.subheader("Extracted Text")
        st.write(extracted_text)

        found_medicines = [med for med in MEDICINE_INVENTORY if med.lower() in extracted_text.lower()]
        if found_medicines:
            st.subheader("Matched Medicines in Inventory")
            order_details = []
            for med in found_medicines:
                st.markdown(f"*{med}* - Stock: {MEDICINE_INVENTORY[med]['stock']} packs, Dosage: {MEDICINE_INVENTORY[med]['dosage']}")
                qty = st.number_input(f"Enter quantity for {med}:", min_value=0, max_value=MEDICINE_INVENTORY[med]["stock"], value=0)
                if qty > 0:
                    order_details.append((med, qty))
            if st.button("Create Order"):
                if order_details:
                    st.success("Order Created Successfully! 🎉")
                    for med, q in order_details:
                        MEDICINE_INVENTORY[med]["stock"] -= q
                    for med, q in order_details:
                        st.write(f"{med} x {q} packs | Remaining stock: {MEDICINE_INVENTORY[med]['stock']}")
                else:
                    st.warning("No medicines selected.")
        else:
            st.subheader("No Matching Medicines Found")
            st.info("Try updating the inventory.")

# -------------------------------------------
# 4) Diagnostic Assistant
# -------------------------------------------
def diagnosis_assistant():
    st.title("Diagnostic Assistant 🩺")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": INITIAL_RESPONSE}]
    for message in st.session_state.chat_history:
        role = "user" if message["role"] == "user" else "assistant"
        avatar = "🗨️" if role == "user" else "💉"
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Describe your symptoms here...")
    if user_prompt:
        with st.chat_message("user", avatar="🗨️"):
            st.markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        messages = [{"role": "system", "content": CHAT_CONTEXT}, 
                    {"role": "assistant", "content": INITIAL_RESPONSE}] + st.session_state.chat_history

        with st.chat_message("assistant", avatar="💉"):
            stream = client.chat.completions.create(model="llama3-8b-8192", messages=messages, stream=True)
            response_content = "".join([chunk.choices[0].delta.content or "" for chunk in stream])
            st.markdown(response_content)

        st.session_state.chat_history.append({"role": "assistant", "content": response_content})

# -------------------------------------------
# 5) X-ray Analyser
# -------------------------------------------
def xray_analyser():
    st.title("X-ray Analyser 📊")
    st.write("Upload an X-ray image to get a prediction from the trained CNN model.")
    uploaded_xray = st.file_uploader("Upload X-ray Image...", type=["jpg", "jpeg", "png"])
    if uploaded_xray:
        xray_img = PILImage.create(uploaded_xray)
        st.image(xray_img, caption="Uploaded X-ray", width=400)
        preds, _, probs = learn.predict(xray_img)
        threshold = 0.3  # Lowered threshold for sensitive detection
        labels = [learn.dls.vocab[i] for i, p in enumerate(probs) if p > threshold]
        st.subheader("Prediction Results")
        st.write(f"**Predicted Labels:** {', '.join(labels) if labels else 'No significant findings'}")
       # st.write("**Probabilities:**")
        #for i, p in enumerate(probs):
         #   st.write(f"{learn.dls.vocab[i]}: {p:.2f}")

# -------------------------------------------
# 6) Render Selected Feature
# -------------------------------------------
if option == "Home":
    # Home section is already rendered above.
    pass
elif option == "Pharmacist's Assistant":
    pharmacist_assistant()
elif option == "Diagnostic Assistant":
    diagnosis_assistant()
elif option == "X-ray Analyser":
    xray_analyser()
