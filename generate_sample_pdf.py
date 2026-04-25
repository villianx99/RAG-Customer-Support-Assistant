from fpdf import FPDF
import os

def create_sample_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- Title ---
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(200, 15, txt="GlobalTech Solutions", ln=True, align='C')
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Customer Support & Operations Guide", ln=True, align='C')
    pdf.ln(10)
    
    # --- Section 1: Business Hours ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="1. Business Hours & Availability", ln=True, align='L')
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, txt=(
        "Our customer support team is available during the following hours:\n"
        "- Monday to Friday: 9:00 AM to 6:00 PM EST\n"
        "- Saturday: 10:00 AM to 4:00 PM EST\n"
        "- Sunday: Closed\n\n"
        "Please note that support is unavailable during public holidays in the United States."
    ))
    pdf.ln(5)
    
    # --- Section 2: Contact Information ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="2. Official Contact Channels", ln=True, align='L')
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, txt=(
        "Primary Email: support@globaltech-solutions.com\n"
        "Phone Support: +1 (800) 123-4567\n"
        "Headquarters: 456 Innovation Drive, Silicon Valley, CA 94025, USA."
    ))
    pdf.ln(5)
    
    # --- Section 3: Refund & Return Policy (ESCALATION TEST) ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="3. Refund and Return Policy", ln=True, align='L')
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, txt=(
        "GlobalTech Solutions offers a 30-day money-back guarantee for most products. "
        "To be eligible for a full refund, the following criteria must be met:\n"
        "- The product must be in its original packaging.\n"
        "- Proof of purchase (receipt or invoice) must be provided.\n"
        "- The request must be submitted within 30 days of the delivery date.\n\n"
        "IMPORTANT: Software licenses and digital downloads are strictly non-refundable "
        "once the product key has been activated. If you face any issues with activation, "
        "please ask for a human agent immediately to resolve the fraud check."
    ))
    pdf.ln(5)
    
    # --- Section 4: Limited Warranty ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="4. Product Warranty", ln=True, align='L')
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, txt=(
        "All hardware products come with a 1-year limited warranty covering manufacturing "
        "defects. This warranty does not cover accidental damage, theft, or unauthorized "
        "modifications. For repairs, customers must ship the product to our Silicon Valley "
        "service center at their own expense."
    ))
    
    # --- Save File ---
    output_path = "GlobalTech_Support_Guide.pdf"
    pdf.output(output_path)
    print(f"Successfully created: {output_path}")

if __name__ == "__main__":
    create_sample_pdf()
