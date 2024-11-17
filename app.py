import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from fpdf import FPDF
import tempfile
from PIL import Image as PILImage

# Load the trained model
model_path = 'AlzModel.h5'
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# Function to preprocess the image
def preprocess_image(img):
    # Resize to target size (244, 244) as done during training
    img = img.resize((244, 244))
    
    # Convert to array and expand dimensions
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize input
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Function to generate PDF
# Function to generate PDF
# Function to generate gender-specific precautions
def get_precautions(result, gender):
    precautions = {
        'Mild Demented': {
            'Male': [
                "Engage in physical exercises such as walking or yoga to maintain mobility and cognitive function.",
                "Include brain-stimulating activities like chess or puzzles.",
                "Ensure regular health check-ups, especially for heart health.",
                "Consume a balanced diet rich in omega-3 fatty acids and antioxidants.",
                "Maintain social interactions to reduce the risk of depression."
            ],
            'Female': [
                "Practice mindfulness and meditation to reduce stress.",
                "Engage in light aerobic activities like swimming or gardening.",
                "Focus on calcium and vitamin D intake to support bone health.",
                "Take part in creative hobbies such as painting or music therapy.",
                "Stay socially active through clubs, volunteer work, or family gatherings."
            ]
        },
        'Moderate Demented': {
            'Male': [
                "Follow a structured daily routine to reduce confusion.",
                "Use assistive devices like reminder clocks for medications.",
                "Avoid alcohol and smoking to prevent further cognitive decline.",
                "Have regular supervision to ensure safety.",
                "Ensure proper hydration and frequent meals to maintain energy."
            ],
            'Female': [
                "Create a calm and predictable environment at home.",
                "Encourage participation in activities like knitting or journaling.",
                "Support with memory aids such as labeled cupboards and calendars.",
                "Avoid overstimulation by limiting loud noises or clutter.",
                "Provide emotional support and companionship to ease anxiety."
            ]
        },
        'Non Demented': {
            'Male': [
                "Continue a healthy lifestyle with regular exercise and a nutritious diet.",
                "Stay engaged in cognitive tasks such as reading or attending workshops.",
                "Monitor for early signs of memory issues during regular check-ups.",
                "Reduce stress by practicing hobbies or relaxation techniques.",
                "Ensure sufficient sleep and a consistent sleep schedule."
            ],
            'Female': [
                "Focus on maintaining a balanced diet with ample fruits and vegetables.",
                "Join community groups or clubs to enhance social engagement.",
                "Incorporate brain-training exercises like Sudoku or crosswords.",
                "Stay physically active with exercises suitable for your age.",
                "Visit healthcare providers regularly for overall health assessment."
            ]
        },
        'Very Mild Demented': {
            'Male': [
                "Use daily planners or apps to stay organized.",
                "Encourage hobbies like woodworking or gardening to remain active.",
                "Limit caffeine and sugar intake to improve focus.",
                "Ensure safety measures such as handrails and good lighting at home.",
                "Encourage open communication with family about any concerns."
            ],
            'Female': [
                "Involve in group therapies or discussions to share experiences.",
                "Participate in light household activities to maintain a sense of purpose.",
                "Keep a journal to track daily tasks and reminders.",
                "Focus on hydration and nutritious meals tailored to your needs.",
                "Provide opportunities for relaxation, such as aromatherapy."
            ]
        }
    }
    return precautions.get(result, {}).get(gender, [])

# Updated PDF generation function
def generate_pdf(name, age, gender, phone_number, result, uploaded_image):
    pdf = FPDF()
    pdf.add_page()
    
    # Set title font (bold, larger size)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Alzheimer's Disease Prediction Report", ln=True, align='C')
    
    # Add a black border around the page
    pdf.set_line_width(1)
    pdf.set_draw_color(0, 0, 0)  # Black color
    pdf.rect(5, 5, 200, 287)  # Adjust dimensions for border

    pdf.ln(10)  # Add a blank line

    # Set font for user info (normal)
    pdf.set_font("Arial", size=12)
    
    # Add user details
    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"Gender: {gender}", ln=True)
    pdf.cell(200, 10, txt=f"Phone Number: {phone_number}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {result}", ln=True)

    # Add precautions to the report
    pdf.ln(10)  # Add some space
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Precautions:", ln=True)

    precautions = get_precautions(result, gender)
    pdf.set_font("Arial", size=12)
    for i, precaution in enumerate(precautions, start=1):
        pdf.cell(200, 10, txt=f"{i}. {precaution}", ln=True)

    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_img_file:
        uploaded_image.save(temp_img_file, format='PNG')
        temp_img_path = temp_img_file.name
        
        # Resize the image to fit the page (max width 180mm)
        img = PILImage.open(temp_img_path)
        img_width, img_height = img.size
        max_width = 90  # Max width of the image in mm (fits the page)
        aspect_ratio = img_height / img_width
        new_height = max_width * aspect_ratio
        img = img.resize((max_width, int(new_height)))

        # Save the resized image to a temporary file
        resized_temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
        img.save(resized_temp_path)

        # Add resized image to PDF (adjust y position as needed)
        pdf.ln(10)  # Add some space before the image
        pdf.image(resized_temp_path, x=10, y=pdf.get_y(), w=max_width)

    # Save the PDF to a file
    report_path = "Alzheimer_Report_with_Image.pdf"
    pdf.output(report_path)
    return report_path

    pdf = FPDF()
    pdf.add_page()
    
    # Set title font (bold, larger size)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Alzheimer's Disease Prediction Report", ln=True, align='C')
    
    # Add a black border around the page
    pdf.set_line_width(1)
    pdf.set_draw_color(0, 0, 0)  # Black color
    pdf.rect(5, 5, 200, 287)  # Adjust dimensions for border

    pdf.ln(10)  # Add a blank line

    # Set font for user info (normal)
    pdf.set_font("Arial", size=12)
    
    # Add user details
    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"Gender: {gender}", ln=True)
    pdf.cell(200, 10, txt=f"Phone Number: {phone_number}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {result}", ln=True)
    pdf.ln(10)

    # Add precautions
    precautions = precautions_dict.get(result, ["No specific precautions available."])
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Precautions:", ln=True)
    pdf.set_font("Arial", size=12)
    for i, precaution in enumerate(precautions, 1):
        pdf.cell(200, 10, txt=f"{i}. {precaution}", ln=True)

    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_img_file:
        uploaded_image.save(temp_img_file, format='PNG')
        temp_img_path = temp_img_file.name
        
        # Resize the image to fit the page (max width 180mm)
        img = PILImage.open(temp_img_path)
        img_width, img_height = img.size
        max_width = 90  # Max width of the image in mm (fits the page)
        aspect_ratio = img_height / img_width
        new_height = max_width * aspect_ratio
        img = img.resize((max_width, int(new_height)))

        # Save the resized image to a temporary file
        resized_temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
        img.save(resized_temp_path)

        # Calculate the x position to center the image
        x_position = (210 - max_width) / 2  # Center the image horizontally

        # Add resized image to PDF (adjust y position as needed)
        pdf.ln(10)  # Add some space before the image
        pdf.image(resized_temp_path, x=x_position, y=pdf.get_y(), w=max_width)

    # Save the PDF to a file
    report_path = "Alzheimer_Report_with_Image.pdf"
    pdf.output(report_path)
    return report_path

    pdf = FPDF()
    pdf.add_page()
    
    # Set title font (bold, larger size)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Alzheimer's Disease Prediction Report", ln=True, align='C')
    
    # Add a black border around the page
    pdf.set_line_width(1)
    pdf.set_draw_color(0, 0, 0)  # Black color
    pdf.rect(5, 5, 200, 287)  # Adjust dimensions for border

    pdf.ln(10)  # Add a blank line

    # Set font for user info (normal)
    pdf.set_font("Arial", size=12)
    
    # Add user details
    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"Gender: {gender}", ln=True)
    pdf.cell(200, 10, txt=f"Phone Number: {phone_number}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {result}", ln=True)

    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_img_file:
        uploaded_image.save(temp_img_file, format='PNG')
        temp_img_path = temp_img_file.name
        
        # Resize the image to fit the page (max width 180mm)
        img = PILImage.open(temp_img_path)
        img_width, img_height = img.size
        max_width = 90  # Max width of the image in mm (fits the page)
        aspect_ratio = img_height / img_width
        new_height = max_width * aspect_ratio
        img = img.resize((max_width, int(new_height)))

        # Save the resized image to a temporary file
        resized_temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
        img.save(resized_temp_path)

        # Calculate the x position to center the image
        x_position = (210 - max_width) / 2  # Center the image horizontally

        # Add resized image to PDF (adjust y position as needed)
        pdf.ln(10)  # Add some space before the image
        pdf.image(resized_temp_path, x=x_position, y=pdf.get_y(), w=max_width)

    # Save the PDF to a file
    report_path = "Alzheimer_Report_with_Image.pdf"
    pdf.output(report_path)
    return report_path

    pdf = FPDF()
    pdf.add_page()
    
    # Set title font (bold, larger size)
    pdf.set_font("Montserrat", "B", 16)
    pdf.cell(200, 10, txt="Alzheimer's Disease Prediction Report", ln=True, align='C')
    
    # Add a black border around the page
    pdf.set_line_width(1)
    pdf.set_draw_color(0, 0, 0)  # Black color
    pdf.rect(5, 5, 200, 287)  # Adjust dimensions for border

    pdf.ln(10)  # Add a blank line

    # Set font for user info (normal)
    pdf.set_font("Poppins", size=12)
    
    # Add user details
    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"Gender: {gender}", ln=True)
    pdf.cell(200, 10, txt=f"Phone Number: {phone_number}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {result}", ln=True)

    # Add the MRI image to the PDF
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_img_file:
        uploaded_image.save(temp_img_file, format='PNG')
        temp_img_path = temp_img_file.name
        
        # Resize the image to fit the page (max width 180mm)
        img = PILImage.open(temp_img_path)
        img_width, img_height = img.size
        max_width = 90  # Max width of the image in mm (fits the page)
        aspect_ratio = img_height / img_width
        new_height = max_width * aspect_ratio
        img = img.resize((max_width, int(new_height)))

        # Save the resized image to a temporary file
        resized_temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
        img.save(resized_temp_path)

        # Add resized image to PDF (adjust y position as needed)
        pdf.ln(10)  # Add some space before the image
        pdf.image(resized_temp_path, x=10, y=pdf.get_y(), w=max_width)

    # Save the PDF to a file
    report_path = "Alzheimer_Report_with_Image.pdf"
    pdf.output(report_path)
    return report_path

# Streamlit app
st.title("Alzheimer's Disease Prediction")
st.write("Provide your details and upload an MRI image to predict the stage of Alzheimer's Disease.")

# User information input fields
name = st.text_input("Name")
age = st.number_input("Age", min_value=0, max_value=120, step=1)
gender = st.radio("Gender", ("Male", "Female", "Other"))
phone_number = st.text_input("Phone Number", max_chars=15, placeholder="Enter a valid phone number")

# File uploader widget
uploaded_file = st.file_uploader("Upload an MRI scan...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    if name and age > 0 and gender and phone_number:
        # Load and display image
        img = image.load_img(uploaded_file)
        st.image(img, caption="Uploaded MRI Scan", use_container_width=True)

        # Preprocess image and make prediction
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)

        # Get prediction result
        predicted_class_index = np.argmax(predictions)
        result = class_labels[predicted_class_index]

        # Show prediction result along with user information
        st.write(f"**Name:** {name}")
        st.write(f"**Age:** {age}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Phone Number:** {phone_number}")
        st.write(f"**Prediction:** {result}")

        # Generate report and automatically trigger download
        if st.button("Download Report"):
            report_path = generate_pdf(name, age, gender, phone_number, result, img)
            with open(report_path, "rb") as file:
                st.download_button(
                    label="Download Report as PDF",
                    data=file,
                    file_name="Alzheimer_Report_with_Image.pdf",
                    mime="application/pdf",
                    key="report_download",
                )
    else:
        st.warning("Please provide all the required information.")
