import cv2
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import pandas as pd
import re
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import requests
import base64
from PyPDF2 import PdfReader


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to convert PDF to images
def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images

# Function to preprocess images for better OCR accuracy
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(binary, None, 30, 7, 21)
    
    return denoised


# Function to extract text from images using Tesseract OCR
def ocr_text(image):
    text = pytesseract.image_to_string(image, config='--psm 6')
    return text


# Function to extract mathematical expressions using MathPix API
def ocr_math(image, app_id, app_key):
    _, encoded_image = cv2.imencode('.png', image)
    b64_image = base64.b64encode(encoded_image).decode('utf-8')

    headers = {
        'app_id': app_id,
        'app_key': app_key,
        'Content-Type': 'application/json'
    }
    data = {
        'src': f'data:image/png;base64,{b64_image}',
        'formats': ['text']
    }

    response = requests.post('https://api.mathpix.com/v3/text', json=data, headers=headers)
    response_data = response.json()

    return response_data.get('text', '')

# Function to combine text and mathematical OCR results
def combine_text_and_math(text, math_text):
    combined_text = f"{text}\n{math_text}"
    return combined_text

# Main function to process PDF and extract text
def process_pdf(pdf_path, app_id, app_key):
    images = pdf_to_images(pdf_path)
    results = []

    for image in images:
        preprocessed_image = preprocess_image(image)
        text = ocr_text(preprocessed_image)
        math_text = ocr_math(preprocessed_image, app_id, app_key)
        combined_text = combine_text_and_math(text, math_text)
        results.append(combined_text)

    return results


# # Function to parse the extracted text and convert it to a DataFrame
def parse_text_to_dataframe(texts):
    data = {
        'Question_ID': [],
        'Skill': [],
        'Correct Answer': [],
        'Question Difficulty': []
    }

    current_section = None
    skill_text = ""

    for text in texts:
        lines = text.splitlines()
        for line in lines:
            if re.match(r"^Question ID\s*\s*", line):
                data['Question_ID'].append(line.split(" ")[-1].strip())
                current_section = 'Question_ID'
            elif re.match(r"^SAT Math\s*\s*", line):
                #data['Skill'].append(line)
                skill_text += line.strip() + " "
                current_section = 'Skill'
            elif re.match(r"^Correct\s*Answer\s*:\s*", line):
                data['Correct Answer'].append(line.split(": ")[-1].strip())
                current_section = 'Correct Answer'
            elif re.match(r"^Question\s*Difficulty\s*:\s*", line):
                data['Question Difficulty'].append(line.split(": ")[-1].strip())
                current_section = 'Question Difficulty'
            else:
                if current_section == 'Skill':
                    skill_text += line.strip() + " "
                    current_section = 'next'

    data['Skill'].append(skill_text.strip())
    
    # Ensure all lists have the same length by adding empty strings where necessary
    max_length = max(len(data[col]) for col in data)
    for col in data:
        while len(data[col]) < max_length:
            data[col].append("")
            
    return pd.DataFrame(data)


######################## get question image #############################
def find_text_coordinates(pdf_path, search_text):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    # Iterate through each page in the PDF
    for page_number in range(len(pdf_document)):
        # Load the page
        page = pdf_document.load_page(page_number)
        
        # Search for the text
        search_results = page.search_for(search_text)
        
        # If search results are found, print them
        if search_results:
            print(f"Found '{search_text}' on page {page_number + 1} at coordinates: {search_results}")
            return search_results, page_number
    
    # If the text is not found in the document, return None
    print(f"'{search_text}' not found in the document.")
    return None, None


def extract_pdf_section(pdf_path, output_image_path, page_number, rect):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    # Extract the page
    page = pdf_document.load_page(page_number)
    
    # Get the portion of the page as an image
    pix = page.get_pixmap(clip=rect)
    
    # Save the image
    pix.save(output_image_path)

    return pix

######################## get question image #############################




if __name__ == "__main__":
    pdf_path = 'SAT2.pdf'
    app_id = 'your_app_id'
    app_key = 'your_app_key'
    
    # results = process_pdf(pdf_path, app_id, app_key)
    # for result in results:
    #     print(result)

    # Convert PDF to images
    images = pdf_to_images(pdf_path)

    # Extract text from images using Tesseract OCR
    texts = process_pdf(pdf_path, app_id, app_key)

    # Parse the extracted text and convert it to a DataFrame
    df = parse_text_to_dataframe(texts)


    output_image_path = "extracted_question.png"  # Replace with your desired output image path

    # Define the page number and the coordinates for the bounding box where the question is located (x0, y0, x1, y1)
    search_text = "Correct Answer:"
    coordinates, page_number = find_text_coordinates(pdf_path, search_text)
    #page_number = 0  # Replace with the actual page number
    rect = fitz.Rect(5, 166, 590, coordinates[0].y1-50)  # left, top, right, bottom-variable 

    # Call the function
    extract_pdf_section(pdf_path, output_image_path, page_number, rect)   

    # Path to save the CSV file
    csv_path = 'SAT2.csv'

    # Save the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)

    print(f"CSV file saved to {csv_path}")
