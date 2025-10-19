import os
from pypdf import PdfReader

# Folder containing PDFs
pdf_folder = "textbook_create/textbook-pdf"
# Output text file
txt_file = "textbook.txt"

all_text = ""

# Loop through all PDF files in the folder
for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"📘 Processing: {filename}")

        reader = PdfReader(pdf_path)

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + "\n"
            else:
                # Mark pages that may have images only
                all_text += f"[Page {page_num + 1} in {filename} contains images only or unreadable text]\n"

# Save all text to a single file
with open(txt_file, "w", encoding="utf-8") as f:
    f.write(all_text)

print(f"✅ All PDFs in '{pdf_folder}' merged into {txt_file}")
