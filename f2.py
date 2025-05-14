import streamlit as st
from groq import Groq
from SPARQLWrapper import SPARQLWrapper, JSON
import tiktoken
import os
from dotenv import load_dotenv
from onto import get_drug_interactions , get_wikidata_id , visualize_graph
import matplotlib.pyplot as plt
import requests
import xml.etree.ElementTree as ET
import arxiv
from context_med import get_context 
from rag import create_vectorstore_from_text, query_vectorstore 
from context_med import get_wikipedia_documents, get_arxiv_documents
import subprocess
import sys
import webbrowser
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings

# Add this near the top of the file after other imports
if 'user_details' not in st.session_state:
    st.session_state.user_details = None

# Load environment variables
load_dotenv()
key = os.getenv('API_KEY')
mistral_key = os.getenv('MISTRAL_API_KEY')
if not key:
    st.error("API key is missing. Please set the API_KEY in your .env file.")
    st.stop()

client = Groq(api_key=key)

# Maximum file size for upload (in MB)
MAX_FILE_SIZE_MB = 5

# Custom CSS for transparent background image
st.markdown("""
    <style>
        /* Transparent background image spanning the full page */
        .stApp {
            background: url('https://img.freepik.com/free-photo/medicine-capsules-global-health-with-geometric-pattern-digital-remix_53876-126742.jpg') no-repeat center center fixed;
            background-size: cover;
            opacity: 1; /* Adjust transparency */
        }
        .main {
            background-color: rgba(255, 255, 255, 0.85); /* Slightly opaque white background */
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #008CBA; 
            color: white; 
            border-radius: 5px;
        }
        .stTextInput, .stTextArea, .stNumberInput, .stSelectbox {
            border-radius: 5px;
        }
        /* Set text color to black for the main app */
        body, .stApp, .main, .stMarkdown {
            color: black;
        }
        /* Set text color to white for the entire sidebar */
        [data-testid="stSidebar"] * {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)


def calculate_bmi(weight, height):
    """Calculate BMI and return the value along with its category."""
    if height > 0:
        bmi = weight / ((height / 100) ** 2)  # Convert height to meters
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 24.9:
            category = "Normal weight"
        elif 25 <= bmi < 29.9:
            category = "Overweight"
        else:
            category = "Obesity"
        return round(bmi, 2), category
    return None, "Invalid height"

# Modify the enter_details function to store data in session state
def enter_details():
    """Collect user details with better UI design."""
    st.header("🩺 Patient Information")
    with st.expander("Step 1: Fill in your basic details 👇", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name:")
            age = st.number_input("Age:", min_value=0, max_value=120, step=1)
            gender = st.radio("Gender:", ["Male", "Female"], horizontal=True)
        with col2:
            phone_number = st.text_input("Phone Number:")
            email = st.text_input("Email Address:")

        weight = st.number_input("Weight (kg):", min_value=0.0, max_value=500.0, step=0.1)
        height = st.number_input("Height (cm):", min_value=0.0, max_value=300.0, step=0.1)

        if weight > 0 and height > 0:
            bmi, category = calculate_bmi(weight, height)
            st.metric(label="Your BMI", value=f"{bmi} ({category})")

    with st.expander("Step 2: Fill in your medical history 👇"):
        allergies = st.text_area("Allergies:")
        medications = st.text_area("Ongoing Medications (Include duration and medicine name):")
        tests = st.text_area("Tests undergone (Include last tested metric):")

    with st.expander("Step 3: Enter current diagnosis 👇"):
        current_disease = st.text_input("Current Disease Detected:")
        medical_tests = st.text_area("Medical Tests Done:")
        hospital_name = st.text_input("Name of Hospital:")
        doctor_qualification = st.text_input("Doctor's Qualification:")
        prescribed_medicines = st.text_area("Medicines Prescribed:")

    if st.button("✅ Submit Details"):
        user_data = {
            "name": name, "age": age, "gender": gender, "phone": phone_number, "email": email,
            "weight": weight, "height": height, "bmi": bmi, "bmi_category": category,
            "allergies": allergies, "medications": medications, "tests": tests,
            "current_disease": current_disease, "medical_tests": medical_tests,
            "hospital_name": hospital_name, "doctor_qualification": doctor_qualification,
            "prescribed_medicines": prescribed_medicines
        }
        st.session_state.user_details = user_data  # Store in session state
        return user_data
    return st.session_state.user_details  # Return from session state if exists

def patient_feedback():
    """Collect patient feedback for deeper analysis."""
    st.header("📝 Patient Feedback")
    with st.expander("Step 4: Provide your feedback 👇", expanded=True):
        feedback = st.text_area("Share your feedback about the diagnosis and treatment:")
        additional_results = st.text_area("Provide any additional results or observations:")
        if st.button("Submit Feedback"):
            st.success("✅ Feedback submitted successfully!")
            return {"feedback": feedback, "additional_results": additional_results}
    return None




def fetch_drug_context(drugs):
    """Fetch combined context for multiple drugs with error handling."""
    drug_contexts = []
    for drug in drugs.split(","):
        drug = drug.strip()  # Remove any extra whitespace
        if not drug:
            continue
            
        try:
            # Get Wikidata context
            try:
                drug_id = get_wikidata_id(drug)
                wikidata_context = get_drug_interactions(drug_id, drug)
            except Exception as e:
                st.warning(f"⚠️ Could not fetch Wikidata information for {drug}: {str(e)}")
                wikidata_context = "No Wikidata information available."

            # Get additional context with retries
            max_retries = 3
            retry_count = 0
            context_text = None
            
            while retry_count < max_retries:
                try:
                    context_text = get_context(drug)
                    break
                except requests.exceptions.ConnectionError:
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(2)  # Wait 2 seconds before retrying
                    else:
                        st.warning(f"⚠️ Could not connect to medical databases for {drug}. Using limited information.")
                        context_text = "Medical database information temporarily unavailable."
                except Exception as e:
                    st.warning(f"⚠️ Error fetching additional context for {drug}: {str(e)}")
                    context_text = "Additional context unavailable."
                    break

            # Combine available information
            drug_contexts.append(f"""
            Drug: {drug}
            
            Wikidata Information:
            {wikidata_context}
            
            Additional Context:
            {context_text}
            """)
            
        except Exception as e:
            st.error(f"❌ Error processing drug {drug}: {str(e)}")
            drug_contexts.append(f"Drug: {drug}\nError: Could not fetch information.")

    # Return combined context or fallback message
    if drug_contexts:
        return "\n\n".join(drug_contexts)
    return "No drug information available. Please check your internet connection and try again."



def fetch_disease_context(disease):
    context = get_context(disease)
    return context

def analyze_data(user_details):
    """Fetch context and generate a patient report."""
    with st.spinner("🔍 Analyzing Data..."):

        # Fetch drug-specific context
        drug_context = fetch_drug_context(user_details["medications"]) + fetch_drug_context(user_details["prescribed_medicines"])
        disease_context = fetch_disease_context(user_details["current_disease"])

        # Combine all user details and additional context
        combined_context = f"""
        Patient Details:
        Name: {user_details['name']}
        Age: {user_details['age']}
        Gender: {user_details['gender']}
        Phone: {user_details['phone']}
        Email: {user_details['email']}
        Weight: {user_details['weight']} kg
        Height: {user_details['height']} cm
        BMI: {user_details['bmi']} ({user_details['bmi_category']})

        Medical History:
        Allergies: {user_details['allergies']}
        Medications: {user_details['medications']}
        Tests: {user_details['tests']}

        Current Diagnosis:
        Disease Detected: {user_details['current_disease']}
        Medical Tests Done: {user_details['medical_tests']}
        Hospital Name: {user_details['hospital_name']}
        Doctor's Qualification: {user_details['doctor_qualification']}
        Prescribed Medicines: {user_details['prescribed_medicines']}

        Feedback:
        {user_details.get('feedback', 'No feedback provided.')}

        Drug-Specific Context:{drug_context}
        Disease-Specific Context:{disease_context}
        """

        # Debugging: Display the combined context being sent to the LLM
        st.subheader("Context Sent to LLM:")
        st.code(combined_context, language="markdown")

        # Prompt for the LLM
        prompt = f"""
        You are a pharmacovigilance expert. Based on the following patient details and context, analyze the potential for adverse drug reactions (ADRs) and provide detailed insights and recommendations:
        {combined_context}
        """

        # Call the LLM to generate the report
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a pharmacovigilance expert."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile"
        )

        # Extract the LLM's response
        insights = response.choices[0].message.content

    # Display the report to the user
    st.success("✅ ADR Analysis Report Generated Successfully!")
    st.subheader("Adverse Drug Reaction Analysis Report 📝")
    st.write(insights)

    return insights  # Return insights for further use



def analyze_drug_interaction(user_details):
    """Visualize the interaction graph for each drug in the medications and prescribed medicines and display it in Streamlit."""
    st.header("🔗 Drug Interaction Visualization")
    medications = user_details["medications"] + "," + user_details["prescribed_medicines"]  # Combine medications and prescribed medicines
    for drug in medications.split(","):
        drug = drug.strip()  # Remove any extra whitespace
        if drug:  # Ensure the drug name is not empty
            try:
                drug_id = get_wikidata_id(drug)
                interactions = get_drug_interactions(drug_id, drug)
                if interactions:
                    plot = visualize_graph(interactions, drug)
                    st.pyplot(plot)  # Display the plot in Streamlit
                else:
                    st.warning(f"No interactions found for {drug}.")
            except ValueError as e:
                st.error(f"Error fetching data for {drug}: {e}")


# Add these imports at the top


# Add after other session state initializations
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

def clean_text(text):
    """Clean text by removing special characters and normalizing whitespace."""
    import re
    import unicodedata
    
    # Normalize unicode characters
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) not in ['So', 'Cn'])
    
    # Replace smart quotes and dashes
    text = text.replace('"', '"').replace('"', '"').replace('—', '-')
    
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text

def initialize_vectorstore(medications):
    """Initialize vectorstore for the FAQ bot."""
    try:
        if st.session_state.vectorstore is None:
            with st.spinner("Initializing knowledge base..."):
                all_texts = []
                for med in medications:
                    print(f"\nFetching documents for: {med}")
                    wiki_docs = get_wikipedia_documents(med)
                    time.sleep(1)  # Prevent rate limiting
                    arxiv_docs = get_arxiv_documents(med)
                    
                    if wiki_docs or arxiv_docs:
                        cleaned_texts = []
                        for doc in wiki_docs + arxiv_docs:
                            cleaned_text = clean_text(doc.page_content)
                            if cleaned_text.strip():
                                cleaned_texts.append(cleaned_text)
                        
                        if cleaned_texts:
                            all_texts.extend(cleaned_texts)
                
                if all_texts:
                    full_text = "\n\n".join(all_texts)
                    
                    try:
                        embeddings = MistralAIEmbeddings(
                            api_key=mistral_key,
                            model="mistral-embed"
                        )
                        
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=50,
                            separators=["\n\n", "\n", ". ", " ", ""]
                        )
                        
                        chunks = text_splitter.split_text(full_text)
                        
                        persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
                        os.makedirs(persist_dir, exist_ok=True)
                        
                        st.session_state.vectorstore = Chroma.from_texts(
                            texts=chunks,
                            embedding=embeddings,
                            persist_directory=persist_dir
                        )
                        return True
                    except Exception as ve:
                        st.error(f"Vectorstore error: {str(ve)}")
                        return False
                else:
                    st.warning("No documents found for the medications.")
                    return False
    except Exception as e:
        st.error(f"Error initializing knowledge base: {str(e)}")
        return False

def get_bot_response(query):
    """Generate response using RAG."""
    try:
        if st.session_state.vectorstore is None:
            return "Knowledge base not initialized. Please wait..."
            
        docs = query_vectorstore(query, st.session_state.vectorstore)
        
        if not docs:
            return "I couldn't find any relevant information for your query."
            
        context = "\n".join([doc.page_content for doc in docs])
        print(f"Context for query '{query}': {context}")
        prompt = f"""
        You are a medical expert. Based on the following context, provide a clear and accurate answer to the user's question.
        If you're unsure about something, please say so.
        
        Context:
        {context}
        
        Question:
        {query}
        """
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a medical expert."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile"
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def show_faq_bot():
    """Display the FAQ bot interface."""
    st.header("💬 Medical Knowledge Assistant")
    
    # Initialize vectorstore if needed
    if st.session_state.vectorstore is None:
        medications = []
        if st.session_state.user_details:
            current_meds = st.session_state.user_details.get("medications", "").strip()
            prescribed_meds = st.session_state.user_details.get("prescribed_medicines", "").strip()
            
            if current_meds:
                medications.extend(med.strip() for med in current_meds.split(","))
            if prescribed_meds:
                medications.extend(med.strip() for med in prescribed_meds.split(","))
            
            medications = [med for med in medications if med]
            
            if medications:
                initialize_vectorstore(medications)
            else:
                st.warning("No medications found. Please enter medications in the form.")
                return
        else:
            st.warning("Please fill out the patient information first.")
            return
    
    # Display chat interface
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask your medical question here..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_bot_response(prompt)
                st.markdown(response)
                st.session_state.chat_messages.append({"role": "assistant", "content": response})

def show_medical_assistant():
    """Display the Medical Assistant interface with proper chat input placement."""
    st.header("💬 Medical Assistant")
    
    # Initialize vectorstore if needed
    if st.session_state.vectorstore is None:
        medications = []
        if st.session_state.user_details:
            current_meds = st.session_state.user_details.get("medications", "").strip()
            prescribed_meds = st.session_state.user_details.get("prescribed_medicines", "").strip()
            
            if current_meds or prescribed_meds:
                medications = [med.strip() for med in (current_meds + "," + prescribed_meds).split(",") if med.strip()]
                if medications:
                    with st.spinner("Initializing medical knowledge base..."):
                        initialize_vectorstore(medications)
                        st.session_state.show_chat = True
            else:
                st.warning("No medications found. Please enter medications in the form.")
                return
        else:
            st.warning("Please fill out the patient information first.")
            return

    # Chat interface
    if st.session_state.show_chat and st.session_state.vectorstore is not None:
        # Display chat history
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_messages = []
            st.rerun()

        # Chat input at root level
        prompt = st.chat_input("Ask your medical question here...")
        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_bot_response(prompt)
                    st.markdown(response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})


def main():
    """Main Streamlit app function with persistent reports."""
    st.title("🏥 Adverse DRUG Interactions")

    # Initialize all session states
    if 'user_details' not in st.session_state:
        st.session_state.user_details = None
    if 'analysis_report' not in st.session_state:
        st.session_state.analysis_report = None
    if 'drug_interactions_data' not in st.session_state:
        st.session_state.drug_interactions_data = {}
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Patient Information"
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    # Sidebar navigation
    st.sidebar.title("Navigation")
    selected_tab = st.sidebar.radio(
        "Go to",
        ["Patient Information", "Analysis Report", "Drug Interactions", "Medical Assistant", "Feedback"]
    )
    st.session_state.active_tab = selected_tab

    # Main content area
    if st.session_state.active_tab == "Patient Information":
        st.header("🩺 Patient Information")
        user_details = enter_details()
        if user_details:
            st.success("✅ Basic details submitted successfully!")
            st.session_state.user_details = user_details

    elif st.session_state.active_tab == "Analysis Report" and st.session_state.user_details:
        st.header("📊 Analysis Report")
        
        # Create two columns for buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate Report Button
            if st.button("🔄 Generate New Report", key="generate_report"):
                with st.spinner("Analyzing data..."):
                    report = analyze_data(st.session_state.user_details)
                    st.session_state.analysis_report = report
                    st.success("✅ New report generated successfully!")
        
        with col2:
            # View Report Button - Always enabled if report exists
            if st.button("👁️ View Latest Report", key="view_report", disabled=not st.session_state.analysis_report):
                pass  # No need for additional logic here
        
        # Always display the report if it exists
        if st.session_state.analysis_report:
            st.success("✅ ADR Analysis Report")
            st.markdown(st.session_state.analysis_report)
            st.info("💡 This is your latest generated report.")

    elif st.session_state.active_tab == "Drug Interactions" and st.session_state.user_details:
        st.header("🔗 Drug Interactions")
        
        # Store and display drug interactions
        button_label = "Regenerate Drug Interactions" if st.session_state.drug_interactions_data else "Visualize Drug Interactions"
        if st.button(button_label):
            with st.spinner("Generating visualizations..."):
                st.session_state.drug_interactions_data = {}  # Clear previous data
                medications = st.session_state.user_details["medications"] + "," + st.session_state.user_details["prescribed_medicines"]
                
                for drug in medications.split(","):
                    drug = drug.strip()
                    if drug:
                        try:
                            drug_id = get_wikidata_id(drug)
                            interactions = get_drug_interactions(drug_id, drug)
                            if interactions:
                                st.session_state.drug_interactions_data[drug] = {
                                    'interactions': interactions,
                                    'plot': visualize_graph(interactions, drug)
                                }
                        except Exception as e:
                            st.error(f"Error processing {drug}: {str(e)}")
        
        # Display stored interactions
        if st.session_state.drug_interactions_data:
            for drug, data in st.session_state.drug_interactions_data.items():
                st.subheader(f"Interactions for {drug}")
                st.pyplot(data['plot'])

    elif st.session_state.active_tab == "Medical Assistant" and st.session_state.user_details:
        st.header("💬 Medical Assistant")
        
        # Initialize medical assistant if needed
        if st.session_state.vectorstore is None:
            medications = []
            current_meds = st.session_state.user_details.get("medications", "").strip()
            prescribed_meds = st.session_state.user_details.get("prescribed_medicines", "").strip()
            
            if current_meds or prescribed_meds:
                medications = [med.strip() for med in (current_meds + "," + prescribed_meds).split(",") if med.strip()]
                if medications:
                    with st.spinner("Initializing medical knowledge base..."):
                        initialize_vectorstore(medications)

        # Display chat history
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_messages = []
            st.rerun()

        # Chat input
        if st.session_state.vectorstore is not None:
            if prompt := st.chat_input("Ask your medical question here..."):
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = get_bot_response(prompt)
                        st.markdown(response)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})

    elif st.session_state.active_tab == "Feedback" and st.session_state.user_details:
        feedback = patient_feedback()
        if feedback:
            st.info("Your feedback will be used for deeper analysis.")
            st.session_state.user_details.update(feedback)
            st.success("✅ Thank you for your feedback!")

    else:
        if not st.session_state.user_details:
            st.warning("Please complete the patient information first.")

    # Reset button in sidebar with confirmation
    if st.sidebar.button("🔄 Reset Application"):
        confirm = st.sidebar.warning("Are you sure you want to reset? All data will be lost.")
        if confirm:
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()