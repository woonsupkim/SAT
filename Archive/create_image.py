from PyPDF2 import PdfReader
from PIL import Image
import fitz  # PyMuPDF

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

# Define the paths
pdf_path = "SAT2.pdf"  # Replace with your PDF path
output_image_path = "extracted_rationale.png"  # Replace with your desired output image path

# Define the page number and the coordinates for the bounding box where the question is located (x0, y0, x1, y1)
search_text1 = "Rationale"
search_text2 = "Question Difficulty"
coordinates1, page_number = find_text_coordinates(pdf_path, search_text1)
coordinates2, page_number = find_text_coordinates(pdf_path, search_text2)
#page_number = 0  # Replace with the actual page number
rect = fitz.Rect(5, coordinates1[0].y1+10, 600, coordinates2[0].y1-20)  # left, top, right, bottom-variable

# Call the function
extract_pdf_section(pdf_path, output_image_path, page_number, rect)

print(f"Extracted image saved to: {output_image_path}")
