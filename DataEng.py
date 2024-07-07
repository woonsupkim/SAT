import os
import cv2
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import pandas as pd
import re
from PIL import Image
import numpy as np
import requests
import base64
from io import BytesIO

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

# Function to parse the extracted text and convert it to a dictionary
def parse_text_to_dict(texts, skill):
    data = {
        'Question_ID': [],
        'Skill': [],
        'Correct Answer': [],
        'Question Difficulty': [],
        'Question Image': [],
        'Question Image2': [],
        'Rationale Image': [],
        'Rationale Image2': []
    }

    for text in texts:
        lines = text.splitlines()
        current_section = None

        for line in lines:
            if re.match(r"^Question ID\s*\s*", line):
                data['Question_ID'].append(line.split(" ")[-1].strip())
                current_section = 'Question_ID'
            elif re.match(r"^Correct\s*Answer\s*:\s*", line):
                data['Correct Answer'].append(line.split(": ")[-1].strip())
                current_section = 'Correct Answer'
            elif re.match(r"^Question\s*Difficulty\s*:\s*", line):
                data['Question Difficulty'].append(line.split(": ")[-1].strip())
                current_section = 'Question Difficulty'

        data['Skill'].append(skill)

    # Ensure all lists have the same length by adding empty strings where necessary
    max_length = max(len(data[col]) for col in data)
    for col in data:
        while len(data[col]) < max_length:
            data[col].append("")
            
    return data

# Function to find the coordinates of the text in the PDF
def find_text_coordinates(pdf_path, search_text, page_number):
    pdf_document = fitz.open(pdf_path)
    
    page = pdf_document.load_page(page_number)
    search_results = page.search_for(search_text)
    
    if search_results:
        return search_results, page_number
    else:
        return None, None

# Function to extract a section of the PDF as an image
def extract_pdf_section(pdf_path, page_number, rect):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number)
    try:
        # Ensure rectangle coordinates are within page bounds
        rect = fitz.Rect(
            max(0, rect.x0),
            max(0, rect.y0),
            min(page.rect.width, rect.x1),
            min(page.rect.height, rect.y1)
        )
        pix = page.get_pixmap(clip=rect)
        
        # Convert to image
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception as e:
        print(f"Error extracting PDF section: {e}")
        image = None
    
    return image

def image_to_byte_array(image: Image) -> bytes:
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

# Main function to process all PDFs in a folder and save data to a dataframe
def process_all_pdfs_in_folder(folder_path, app_id, app_key):
    all_data = {
        'Question_ID': [],
        'Skill': [],
        'Correct Answer': [],
        'Question Difficulty': [],
        'Question Image2': [],
        'Rationale Image2': []
    }

    for filename in os.listdir(folder_path):
        if filename.endswith('Answers.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing {pdf_path}...")
            
            skill = os.path.splitext(filename)[0]  # Use the file name (without extension) as the skill

            # Process each PDF
            results = process_pdf(pdf_path, app_id, app_key)
            parsed_data = parse_text_to_dict(results, skill)

            for key in all_data:
                all_data[key].extend(parsed_data[key])

            images = pdf_to_images(pdf_path)
            for page_number in range(len(images)):
                search_text0 = ":"
                search_text1 = "Rationale"
                coordinates0, page_number = find_text_coordinates(pdf_path, search_text0, page_number)
                coordinates01, page_number = find_text_coordinates(pdf_path, search_text1, page_number)
                if coordinates0 and coordinates01:
                    rect = fitz.Rect(0, coordinates0[0].y1+8, 842, coordinates01[0].y1-65)
                    question_image = extract_pdf_section(pdf_path, page_number, rect)
                    if question_image:
                        img_bytes = image_to_byte_array(question_image)
                        all_data['Question Image2'].append(img_bytes)
                    else:
                        all_data['Question Image2'].append("")
                
                search_text2 = "Question Difficulty"
                coordinates1, page_number = find_text_coordinates(pdf_path, search_text1, page_number)
                coordinates2, page_number = find_text_coordinates(pdf_path, search_text2, page_number)
                if coordinates1 and coordinates2:
                    rect = fitz.Rect(0, coordinates1[0].y1+10, 842, coordinates2[0].y1-20)
                    rationale_image = extract_pdf_section(pdf_path, page_number, rect)
                    if rationale_image:
                        img_bytes = image_to_byte_array(rationale_image)
                        all_data['Rationale Image2'].append(img_bytes)
                    else:
                        all_data['Rationale Image2'].append("")

    # Ensure all lists have the same length by adding empty strings where necessary
    max_length = max(len(all_data[col]) for col in all_data)
    for col in all_data:
        while len(all_data[col]) < max_length:
            all_data[col].append("")

    # Convert dictionary to DataFrame and save as CSV
    df = pd.DataFrame(all_data)

    # # Drop rows where the 'Question Image2' column is empty or None
    df_cleaned = df.dropna(subset=['Question Image2'])
    df_cleaned = df_cleaned[df_cleaned['Question Image2'].str.strip() != '']

    # # Drop rows where the 'Rationale Image2' column is empty or None
    df_cleaned = df.dropna(subset=['Rationale Image2'])
    df_cleaned = df_cleaned[df_cleaned['Rationale Image2'].str.strip() != '']

    # # Drop rows where the 'Correct Answer' column is empty or None
    df_cleaned = df.dropna(subset=['Correct Answer'])
    df_cleaned = df_cleaned[df_cleaned['Correct Answer'].str.strip() != '']

    # df_cleaned.to_pickle('images_dataframe.pkl')
    # df_cleaned.to_csv('SAT_Question_Bank_Results.csv', index=False)
    df_cleaned.to_csv('SAT_reading.csv', index=False)

if __name__ == "__main__":
    folder_path = r'C:\Users\Woon\Desktop\Columbia\Applied Analytics\Term3\Projects\Personal\SAT\Data\Reading and Writing'
    app_id = 'your_app_id'
    app_key = 'your_app_key'

    process_all_pdfs_in_folder(folder_path, app_id, app_key)
