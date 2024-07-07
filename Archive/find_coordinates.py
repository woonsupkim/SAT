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

# Define the path to the PDF and the search text
pdf_path = "SAT2.pdf"
search_text = "Correct Answer:"

# Call the function
coordinates, page_number = find_text_coordinates(pdf_path, search_text)

# Extract and print the last coordinate (bottom-right y-coordinate)
if coordinates:
    for rect in coordinates:
        print(f"Coordinates on page {page_number + 1}: {rect}")
        last_y_coordinate = rect.y1
        print(f"Last coordinate (bottom-right y-coordinate): {last_y_coordinate}")
else:
    print("Text not found.")
