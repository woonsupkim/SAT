import fitz  # PyMuPDF
import pandas as pd

# Function to extract text from PDF
def extract_text_from_first_page(pdf_path):
    # Open the PDF file
    document = fitz.open(pdf_path)
    # Extract text from the first page
    first_page_text = document[0].get_text()
    document.close()
    return first_page_text

# Function to parse the extracted text and convert it to a DataFrame
def parse_text_to_dataframe(text):
    lines = text.splitlines()
    
    data = {
        'Assessment': [],
        'Test': [],
        'Domain': [],
        'Skill': [],
        'Difficulty': [],
        'Question': [],
        'Correct Answer': [],
        'Rationale': [],
        'Question Difficulty': []
    }

    current_section = None
    question_text = ""

    for line in lines:
        if line.startswith("Assessment"):
            data['Assessment'].append(line.split(": ")[-1].strip())
            current_section = 'Assessment'
        elif line.startswith("Test"):
            data['Test'].append(line.split(": ")[-1].strip())
            current_section = 'Test'
        elif line.startswith("Domain"):
            data['Domain'].append(line.split(": ")[-1].strip())
            current_section = 'Domain'
        elif line.startswith("Skill"):
            data['Skill'].append(line.split(": ")[-1].strip())
            current_section = 'Skill'
        elif line.startswith("Difficulty"):
            data['Difficulty'].append(line.split(": ")[-1].strip())
            current_section = 'Difficulty'
        elif line.startswith("Question ID"):
            question_text = ""
            current_section = 'Question'
        elif line.startswith("Correct Answer"):
            data['Correct Answer'].append(line.split(": ")[-1].strip())
            current_section = 'Correct Answer'
        elif line.startswith("Rationale"):
            data['Rationale'].append(line.split(": ")[-1].strip())
            current_section = 'Rationale'
        elif line.startswith("Question Difficulty"):
            data['Question Difficulty'].append(line.split(": ")[-1].strip())
            current_section = 'Question Difficulty'
        else:
            if current_section == 'Question':
                question_text += line.strip() + " "
            elif current_section:
                data[current_section][-1] += " " + line.strip()

    # Add the final question text
    if question_text:
        data['Question'].append(question_text.strip())
    
    # Ensure all lists have the same length by adding empty strings where necessary
    max_length = max(len(data[col]) for col in data)
    for col in data:
        while len(data[col]) < max_length:
            data[col].append("")
            
    return pd.DataFrame(data)

# Path to the PDF file
pdf_path = 'SAT2.pdf'

# Extract text from the PDF file
text = extract_text_from_first_page(pdf_path)

# Parse the extracted text and convert it to a DataFrame
df = parse_text_to_dataframe(text)

# Path to save the CSV file
csv_path = 'SAT2.csv'

# Save the DataFrame to a CSV file
df.to_csv(csv_path, index=False)

print(f"CSV file saved to {csv_path}")
