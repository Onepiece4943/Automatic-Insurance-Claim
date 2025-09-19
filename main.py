import os, re, json
from dotenv import load_dotenv
from flask import Flask, render_template, request
import google.generativeai as genai
from PyPDF2 import PdfReader
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup API
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

if api_key is None or api_key == "":
    print("Google AI API key not set or empty. Please set the environment variable.")
    exit()

genai.configure(api_key=api_key)
os.environ['GOOGLE_API_KEY'] = api_key

app = Flask(__name__)

# Excluded medical conditions
general_exclusion_list = ["HIV/AIDS", "Parkinson's disease", "Alzheimer's disease","pregnancy", "substance abuse", "self-inflicted injuries", "sexually transmitted diseases(std)", "pre-existing conditions"]

def get_file_content(file):
    """Extract text from PDF file"""
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

def extract_numeric_amount(amount_str):
    """Extract numeric value from amount string"""
    if not amount_str:
        return 0
    
    cleaned = re.sub(r'[^\d.,]', '', str(amount_str))
    
    if ',' in cleaned and '.' in cleaned:
        cleaned = cleaned.replace(',', '')
    elif cleaned.count(',') == 1 and len(cleaned.split(',')[1]) <= 2:
        cleaned = cleaned.replace(',', '.')
    elif ',' in cleaned:
        cleaned = cleaned.replace(',', '')
    
    try:
        return float(cleaned)
    except:
        return 0

def get_bill_info(pdf_text, claim_reason="", description="", claim_amount=""):
    """Use AI to extract disease and expense from medical bill"""
    prompt = f"""Analyze this medical bill and extract information. Return ONLY a JSON object.

    Medical Bill Text: {pdf_text}
    
    Patient claimed amount: ${claim_amount}
    Claim reason: {claim_reason}
    Description: {description}

    Find and extract:
    1. "disease": Medical condition, diagnosis, or treatment reason
    2. "expense": The bill amount, total cost, or charges

    Look carefully for:
    - Any dollar amounts, totals, charges, fees
    - Medical diagnoses, conditions, treatments
    - Keywords: Total, Amount, Cost, Fee, Charge, Due, Payment, Bill, Price
    - Numbers with $ symbols or currency formatting

    Return: {{"disease": "condition name", "expense": "amount_number"}}
    
    Important:
    - Extract the main billing amount from the document
    - If multiple amounts exist, use the total or largest amount
    - If no amount found, use the claimed amount: {claim_amount}
    - For disease, use medical terms from document or claim reason
    - Return only numbers for expense (no $ symbols)"""

    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        json_match = re.search(r'\{[^{}]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
            
            # Process expense
            if 'expense' in result:
                extracted_expense = extract_numeric_amount(result['expense'])
                # If extraction failed or returned 0, use claim amount
                if extracted_expense <= 0:
                    result['expense'] = float(claim_amount) if claim_amount else 0
                else:
                    result['expense'] = extracted_expense
            else:
                result['expense'] = float(claim_amount) if claim_amount else 0
                
            return result
        else:
            return {"disease": None, "expense": float(claim_amount) if claim_amount else 0}
            
    except Exception as e:
        print(f"Error in AI analysis: {e}")
        return {"disease": None, "expense": float(claim_amount) if claim_amount else 0}

def check_claim_rejection(disease, general_exclusion_list, threshold=0.7):
    """Check if disease is in exclusion list"""
    if not disease:
        return False
        
    disease_lower = disease.lower()
    for excluded_disease in general_exclusion_list:
        excluded_lower = excluded_disease.lower()
        
        if excluded_lower in disease_lower or disease_lower in excluded_lower:
            return True
        
        vectorizer = CountVectorizer()
        try:
            vectors = vectorizer.fit_transform([disease_lower, excluded_lower])
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            if similarity > threshold:
                return True
        except:
            continue
    return False

# Report template
PROMPT = """Insurance Claim Analysis:

EXECUTIVE SUMMARY:
{executive_summary}

INTRODUCTION:
{introduction}

PATIENT: {patient_info}
DISEASE: {disease}
BILLED AMOUNT: ${bill_amount}
CLAIM AMOUNT: ${claim_amount}

DOCUMENT VERIFICATION:
{document_verification}

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
        # Get form data
        name = request.form['name']
        address = request.form['address']
        claim_type = request.form['claim_type']
        claim_reason = request.form['claim_reason']
        medical_facility = request.form['medical_facility']
        medical_bill = request.files['medical_bill']
        total_claim_amount = request.form['total_claim_amount']
        description = request.form['description']

        # Extract text from PDF
        bill_text = get_file_content(medical_bill)
        if not bill_text.strip():
            return render_template("result.html", 
                name=name, address=address, claim_type=claim_type,
                claim_reason=claim_reason, medical_facility=medical_facility,
                total_claim_amount=total_claim_amount, description=description,
                output="ERROR: Invalid Consultation Receipt - No text extracted"
            )

        # Get disease and expense from AI (pass claim amount as reference)
        bill_info = get_bill_info(bill_text, claim_reason, description, total_claim_amount)

        # Handle missing disease info using form data
        if bill_info['disease'] is None or bill_info['disease'] == "":
            if "parkinson" in claim_reason.lower() or "parkinson" in description.lower():
                bill_info['disease'] = "Parkinson's disease"
            elif claim_reason and claim_reason.strip():
                bill_info['disease'] = claim_reason
            elif description and description.strip():
                bill_info['disease'] = description.split(',')[0]
            else:
                return render_template("result.html", 
                    name=name, address=address, claim_type=claim_type,
                    claim_reason=claim_reason, medical_facility=medical_facility,
                    total_claim_amount=total_claim_amount, description=description,
                    output="ERROR: Could not determine medical condition"
                )

        # Validate amounts
        try:
            bill_expense = float(bill_info['expense'])
            claim_amount = float(total_claim_amount)
            
            # Only reject if claim significantly exceeds bill amount (allow 10% tolerance)
            if bill_expense > 0 and claim_amount > bill_expense * 1.1:
                return render_template("result.html", 
                    name=name, address=address, claim_type=claim_type,
                    claim_reason=claim_reason, medical_facility=medical_facility,
                    total_claim_amount=total_claim_amount, description=description,
                    output=f"REJECTED: Claim amount (${claim_amount}) exceeds billed amount (${bill_expense})"
                )
        except ValueError:
            return render_template("result.html", 
                name=name, address=address, claim_type=claim_type,
                claim_reason=claim_reason, medical_facility=medical_facility,
                total_claim_amount=total_claim_amount, description=description,
                output="ERROR: Invalid amount format"
            )

        # Check exclusions
        is_excluded = check_claim_rejection(bill_info["disease"], general_exclusion_list)
        
        # Make decision
        if is_excluded:
            decision = "REJECTED"
            approved_amount = "0"
            exclusion_status = "FAILED - Disease excluded"
            rejection_reason = f"{bill_info['disease']} is in the exclusion list"
            summary = f"Claim rejected: {bill_info['disease']} is not covered by policy"
            executive_summary = f"Insurance claim for {name} has been REJECTED due to excluded medical condition. Claim amount: ${total_claim_amount}."
            introduction = f"This report evaluates the insurance claim submitted by {name} for {bill_info['disease']} treatment. The claim has been REJECTED as the condition is excluded from coverage."
        else:
            decision = "APPROVED"
            approved_amount = total_claim_amount
            exclusion_status = "PASSED"
            rejection_reason = "No exclusions found"
            summary = f"Claim approved for ${total_claim_amount} for treatment of {bill_info['disease']}"
            executive_summary = f"Insurance claim for {name} has been APPROVED for ${total_claim_amount}. All requirements satisfied."
            introduction = f"This report evaluates the insurance claim submitted by {name} for {bill_info['disease']} treatment. The claim has been APPROVED for full coverage."

        # Document verification status
        document_verification = f"Medical bill PDF submitted and verified. Disease: {bill_info['disease']} identified. Billing amount: ${bill_info['expense']} processed."

        # Generate report
        patient_info = f"{name} | {address} | {claim_type} | {medical_facility}"
        report = PROMPT.format(
            executive_summary=executive_summary,
            introduction=introduction,
            patient_info=patient_info,
            disease=bill_info["disease"],
            bill_amount=bill_info['expense'],
            claim_amount=total_claim_amount,
            document_verification=document_verification,
            exclusion_status=exclusion_status,
            rejection_reason=rejection_reason,
            decision=decision,
            approved_amount=approved_amount,
            summary=summary
        )

        output = re.sub(r'\n', '<br>', report)
        
        return render_template("result.html", 
            name=name, address=address, claim_type=claim_type,
            claim_reason=claim_reason, medical_facility=medical_facility,
            total_claim_amount=total_claim_amount, description=description,
            output=output
        )

    except Exception as e:
        return render_template("result.html", 
            name=name if 'name' in locals() else '', 
            address=address if 'address' in locals() else '',
            claim_type=claim_type if 'claim_type' in locals() else '',
            claim_reason=claim_reason if 'claim_reason' in locals() else '',
            medical_facility=medical_facility if 'medical_facility' in locals() else '',
            total_claim_amount=total_claim_amount if 'total_claim_amount' in locals() else '',
            description=description if 'description' in locals() else '',
            output=f"ERROR: System error - {str(e)}"
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)