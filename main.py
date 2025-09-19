# Import necessary libraries
import os, re
from dotenv import load_dotenv
from flask import Flask, render_template, request
import google.generativeai as genai
from PyPDF2 import PdfReader
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Get the Google AI API key from the environment variable
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

if api_key is None or api_key == "":
    print("Google AI API key not set or empty. Please set the environment variable.")
    exit()

# Initialize the Google AI client with the API key
genai.configure(api_key=api_key)
os.environ['GOOGLE_API_KEY'] = api_key

# Flask App
app = Flask(__name__)

chat_history = []
general_exclusion_list = ["HIV/AIDS", "Parkinson's disease", "Alzheimer's disease","pregnancy", "substance abuse", "self-inflicted injuries", "sexually transmitted diseases(std)", "pre-existing conditions"]

def get_file_content(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if not text.strip():
            pdf = PdfReader(file)
            for page_num in range(len(pdf.pages)):
                page = pdf.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    
        return text
        
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def get_bill_info(data):
    prompt = """Extract 'disease' and 'expense amount' from medical document. Return JSON: {"disease": "condition", "expense": "amount"}""" + data

    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        json_match = re.search(r'\{[^{}]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"disease": None, "expense": None}
            
    except Exception as e:
        print(f"Error in get_bill_info: {e}")
        return {"disease": None, "expense": None}

def check_claim_rejection(disease, general_exclusion_list, threshold=0.7):
    """Check if disease is in exclusion list"""
    if not disease:
        return False
        
    disease_lower = disease.lower()
    for excluded_disease in general_exclusion_list:
        excluded_lower = excluded_disease.lower()
        # Direct match check
        if excluded_lower in disease_lower or disease_lower in excluded_lower:
            return True
        
        # Similarity check for partial matches
        vectorizer = CountVectorizer()
        try:
            vectors = vectorizer.fit_transform([disease_lower, excluded_lower])
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            if similarity > threshold:
                return True
        except:
            continue
            
    return False

# SHORTENED PROMPT TEMPLATE
PROMPT = """Insurance Claim Analysis:

PATIENT: {patient_info}
DISEASE: {disease}
CLAIM AMOUNT: ${max_amount}

EXCLUSION CHECK: {exclusion_status}
REASON: {rejection_reason}

DECISION: {decision}
APPROVED AMOUNT: ${approved_amount}

SUMMARY: {summary}
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def msg():
    try:
        name = request.form['name']
        address = request.form['address']
        claim_type = request.form['claim_type']
        claim_reason = request.form['claim_reason']
        date = request.form['date']
        medical_facility = request.form['medical_facility']
        medical_bill = request.files['medical_bill']
        total_claim_amount = request.form['total_claim_amount']
        description = request.form['description']

        # Extract text from PDF
        bill_text = get_file_content(medical_bill)
        if not bill_text.strip():
            return render_template("result.html", 
                output="ERROR: Invalid Consultation Receipt - No text extracted"
            )

        # Extract disease and expense
        bill_info = get_bill_info(bill_text)

        if bill_info['disease'] is None or bill_info['expense'] is None:
            return render_template("result.html", 
                output="ERROR: Could not extract disease or expense information"
            )

        # Check amount validity
        try:
            bill_expense = float(bill_info['expense']) if bill_info['expense'] else 0
            claim_amount = float(total_claim_amount)
            
            if bill_expense < claim_amount:
                return render_template("result.html", 
                    output="REJECTED: Claim amount exceeds billed amount"
                )
        except ValueError:
            return render_template("result.html", 
                output="ERROR: Invalid amount format"
            )

        # Check for exclusions - THIS IS THE CRITICAL FIX
        is_excluded = check_claim_rejection(bill_info["disease"], general_exclusion_list)
        
        # Determine claim decision
        if is_excluded:
            decision = "REJECTED"
            approved_amount = "0"
            exclusion_status = "FAILED - Disease excluded"
            rejection_reason = f"{bill_info['disease']} is in the exclusion list"
            summary = f"Claim rejected: {bill_info['disease']} is not covered"
        else:
            decision = "APPROVED"
            approved_amount = total_claim_amount
            exclusion_status = "PASSED"
            rejection_reason = "No exclusions found"
            summary = f"Claim approved for ${total_claim_amount}"

        # Prepare patient info
        patient_info = f"{name} | {address} | {claim_type} | {medical_facility}"

        # Generate short report
        report = PROMPT.format(
            patient_info=patient_info,
            disease=bill_info["disease"],
            max_amount=total_claim_amount,
            exclusion_status=exclusion_status,
            rejection_reason=rejection_reason,
            decision=decision,
            approved_amount=approved_amount,
            summary=summary
        )

        output = re.sub(r'\n', '<br>', report)
        
        return render_template("result.html", output=output)

    except Exception as e:
        return render_template("result.html", 
            output=f"ERROR: System error - {str(e)}"
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)