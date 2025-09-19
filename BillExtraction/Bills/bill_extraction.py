from PyPDF2 import PdfReader
import google.generativeai as genai
import yaml
import json
import os

CONFIG_PATH = r"config.yaml"

with open(CONFIG_PATH) as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
    api_key = data['GEMINI_API_KEY']

def get_pdf_data(fpath):
    text = ""
    pdf = PdfReader(fpath)
    for page_num in range(len(pdf.pages)):
        page = pdf.pages[page_num]
        page_text = page.extract_text()
        text += page_text
    return text

def get_llm():
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    return model

def get_invoice_info_from_llm(data):
    llm = get_llm()
    prompt = f"""Act as an expert in extracting information from medical invoices. You are given with the invoice details of a patient. Go through the given document carefully and extract the 'disease' and the 'expense amount' from the data. Return the data in json format = {{'disease':'','expense':''}}

INVOICE DETAILS: {data}"""
    
    response = llm.generate_content(prompt)
    data = json.loads(response.text)
    
    return data

if __name__ == '__main__':
    bill_folder = "Bills"
    bill_name = "MedicalBill1.pdf"
    
    bill_path = os.path.join(bill_folder, bill_name)
    if not os.path.exists(bill_path):
        print(f"{bill_path} does not exist. Please check the file location")
    else:
        bill_info = get_pdf_data(bill_path)
        invoice_details = get_invoice_info_from_llm(bill_info)
        
        print(f"Disease: {invoice_details['disease']}")
        print(f"Expense: {invoice_details['expense']}")