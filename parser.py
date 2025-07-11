import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Configure the Gemini API with your key
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("🔴 Error: GEMINI_API_KEY not found. Please set it in your .env file.")
    exit()

# --- 1. File Uploading ---
print("Uploading files...")

directory = "fuvest\\2024\\1a Fase"
if not os.path.exists(directory):
    print(f"🔴 Error: The directory '{directory}' does not exist.")
    exit()

pdf_file_paths = [f"{directory}\\prova.pdf", f"{directory}\\gabarito.pdf"]
uploaded_files = []

for file_path in pdf_file_paths:
    try:
        print(f"Uploading {file_path}...")
        # Upload the file and store the file object
        uploaded_file = genai.upload_file(path=file_path, display_name=file_path)
        uploaded_files.append(uploaded_file)
        print(f"✅ Completed upload for: {uploaded_file.display_name}")
    except FileNotFoundError:
        print(f"🔴 Error: The file '{file_path}' was not found.")
        exit()

print("\nAll files uploaded successfully! ✅")


model = genai.GenerativeModel(model_name='gemini-2.5-flash')

prompt_text = """
You are a specialized data extraction API. Your sole function is to process two uploaded PDF files, prova.pdf and gabarito.pdf, and generate a single, precise JSON output.

Constraints:
  - The output MUST be only the raw JSON object.

Instructions:
 - Analyze the prova.pdf to identify the specific exam version (e.g., "Prova V", "Prova K", etc.).
 - Using the identified exam version, locate the corresponding answer key column in the gabarito.pdf.
 - Read both documents to extract all necessary information.
 - If a piece of information is not explicitly available in the documents (like the PDF URL), perform a web search to find the correct data.
 - Populate the following JSON structure exactly as specified.

JSON Output Structure:
{
    "pdf_link": "https://fuvest.br/2024/prova_geral.pdf",
    "data": "2024-01-01",
    "qtd_questoes": 2,
    "opcoes_resposta": ["A", "B", "C", "D", "E"],
    "questoes": {
        "1": {"disciplina": "Matemática", "resposta": "A"},
        "2": {"disciplina": "História", "resposta": "B"}
    }
}

Field Population Rules:
  - pdf_link: Search online and provide the direct URL for the exam.
  - data: Extract the exam date from the documents and format it as YYYY-MM-DD.
  - qtd_questoes: Determine the total count of questions in the exam.
  - opcoes_resposta: This field should be a static array: ["A", "B", "C", "D", "E"].
  - questoes: This must be an object containing entries for every question number (from 1 to the total). For each question:
      - disciplina: Determine the academic discipline based on the question's content in prova.pdf. Use "Interdisciplinar" if it blends multiple distinct fields.
      - resposta: Extract the correct single-letter answer from the matched answer key in gabarito.pdf.
"""

prompt_parts = [
    prompt_text,
]
prompt_parts.extend(uploaded_files) # Add the list of file objects to the prompt

print("\nSending prompt to Gemini...")

try:
    response = model.generate_content(prompt_parts)
    print("\n--- Gemini Response ---")
    print(response.text)
    print("-----------------------\n")
    output_file_path = os.path.join(directory, "data.json")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(response.text)
except Exception as e:
    print(f"🔴 An error occurred while generating content: {e}")