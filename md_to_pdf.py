import os
import re
from fpdf import FPDF
from fpdf.enums import XPos, YPos

def ultra_clean(text):
    # Keep only ASCII printable characters
    return re.sub(r'[^\x20-\x7E]', ' ', text)

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'SMART AI MONITORING - DETAILED DOCUMENTATION', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def convert_md_to_pdf(md_file, pdf_file):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    with open(md_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                pdf.ln(5)
                continue
            
            line = ultra_clean(line)
            if not line.strip(): continue
            
            # Reset X to Left Margin for every block
            pdf.set_x(10)
            
            if line.startswith('###'):
                pdf.set_font("Helvetica", 'B', 12)
                pdf.ln(3)
                pdf.cell(0, 8, line[3:].strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", size=11)
            elif line.startswith('##'):
                pdf.set_font("Helvetica", 'B', 14)
                pdf.ln(5)
                pdf.cell(0, 10, line[2:].strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", size=11)
            elif line.startswith('#'):
                pdf.set_font("Helvetica", 'B', 16)
                pdf.cell(0, 12, line[1:].strip().upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", size=11)
            elif line.startswith('---'):
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                pdf.ln(5)
            elif line.startswith('*'):
                pdf.multi_cell(0, 7, f"- {line[1:].strip()}")
            else:
                pdf.multi_cell(0, 7, line)

    pdf.output(pdf_file)

if __name__ == "__main__":
    try:
        convert_md_to_pdf('DETAILED_DOCUMENTATION.md', 'SMART_AI_MONITORING_DOCS.pdf')
        print("Success!")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
