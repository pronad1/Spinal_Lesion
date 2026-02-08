import pdfplumber
import sys

pdf = pdfplumber.open('version 2.pdf')
for i, page in enumerate(pdf.pages[:12]):
    print(f"\n{'='*60}\nPAGE {i+1}\n{'='*60}\n")
    text = page.extract_text()
    if text:
        # Replace unicode characters with ASCII approximations
        text = text.encode('ascii', 'replace').decode('ascii')
        print(text)
