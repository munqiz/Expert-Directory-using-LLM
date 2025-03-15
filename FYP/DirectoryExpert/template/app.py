from flask import Flask, request, render_template, redirect, url_for, render_template, send_from_directory, send_file, flash, session
import os
import pdfplumber
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
import traceback
import re
from werkzeug.utils import secure_filename
import secrets

# Initialize Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# File directories and configurations
UPLOAD_FOLDER = 'uploaded_pdfs'
EXPERT_INFO_FOLDER = 'expert_info'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXPERT_INFO_FOLDER'] = EXPERT_INFO_FOLDER

# Download nltk resources
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load pre-trained model for semantic similarity 
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 1: Validate if the file is a proper PDF using PdfReader
def is_valid_pdf(file_path):
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)  
            if len(reader.pages) > 0:
                print(f"{file_path} is a valid PDF.")
                return True
            else:
                print(f"{file_path} is not a valid PDF (no pages found).")
                return False
    except Exception as e:
        print(f"PDF validation failed: {e}")
        return False

# Step 2: Extract text from PDF with fallback options
def extract_text_from_pdf(file_path):
    if not is_valid_pdf(file_path):
        return None, "Invalid or corrupted PDF"
    
    text = ''
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        if not text.strip():
            raise ValueError("No text found in the PDF using pdfplumber")
        print(f"Text extracted from {file_path} using pdfplumber.")
        return text, None
    except Exception as e:
        print(f"Error extracting PDF with pdfplumber: {e}")
        traceback.print_exc()

    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ''
        if text.strip():
            print(f"Text extracted from {file_path} using PyPDF2.")
            return text, None
        else:
            return None, "Failed to extract text using both pdfplumber and PyPDF2"
    except Exception as e:
        print(f"Error extracting PDF with PyPDF2: {e}")
        traceback.print_exc()
        return None, f"Error extracting PDF with PyPDF2: {e}"

# Step 3: Clean and process text
def clean_text(text):
    text = text.replace('-', ' ').replace('\n', ' ').lower()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Step 4: Convert text into embeddings using sentence transformers
def get_text_embedding(text):
    try:
        embedding = model.encode(text, convert_to_tensor=True)
        print(f"Generated embedding for text: {text[:100]}")  # Debugging
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Step 5: Summarize the PDF content using LLM (Optional Step)
def summarize_pdf_text(pdf_text):
    return pdf_text[:10000]  # Simple summarization for large PDFs

# Adjust similarity score based on the word count of the document
def adjust_similarity(similarity, word_count, min_words=1200, max_words=15000):
    """
    Adjusts the similarity score to account for word count bias.
    
    Args:
        similarity (float): Original similarity score (0 to 1).
        word_count (int): Word count of the document.
        min_words (int): Minimum word count expected in the dataset.
        max_words (int): Maximum word count expected in the dataset.

    Returns:
        float: Normalized similarity score.
    """
    # Normalize word count to a range of 0.5 to 1.0
    word_weight = (word_count - min_words) / (max_words - min_words)
    word_weight = max(0.5, word_weight)  # Ensure a minimum weight of 0.5 to avoid zeroing out scores

    # Adjust the similarity score by the normalized word count
    return similarity * word_weight

# Step 6: Admin route for uploading PDFs and saving info
import re  # Add this import for sanitizing the expert's name

import re  # Ensure re is imported for sanitizing filenames
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        password = request.form['password']
        if password == 'admin123':
            session['is_admin'] = True
            return redirect(url_for('admin_upload'))
        else:
            flash('Incorrect password. Please try again.')
            return redirect(url_for('admin_login'))

    return render_template('admin_login.html')
@app.route('/admin', methods=['GET', 'POST'])
def admin_upload():
    # Check if the password was validated
    if session.get('is_admin') != True:
        return redirect(url_for('admin_login'))
    if request.method == 'POST':
        # Handle file upload logic as before
        expert_name = request.form['name']
        expert_email = request.form['email']
        expert_faculty = request.form['faculty']

        if 'pdf_file' not in request.files:
            return 'No file uploaded', 400

        pdf_file = request.files['pdf_file']
        if pdf_file and pdf_file.filename.endswith('.pdf'):
            # Sanitize expert's name for valid filename
            sanitized_name = re.sub(r'\W+', '_', expert_name)
            new_filename = f"{sanitized_name}.pdf"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

            # Save PDF
            pdf_file.save(file_path)

            # Extract PDF text (for debugging)
            pdf_text, error = extract_text_from_pdf(file_path)
            if error:
                return f"Failed to extract PDF text: {error}", 400
            print(f"Extracted PDF text: {pdf_text[:100]}...")

            # Save expert info
            if not os.path.exists(app.config['EXPERT_INFO_FOLDER']):
                os.makedirs(app.config['EXPERT_INFO_FOLDER'])

            expert_info = {
                'name': expert_name,
                'email': expert_email,
                'faculty': expert_faculty,
                'pdf': new_filename,
            }

            expert_info_file = os.path.join(app.config['EXPERT_INFO_FOLDER'], f"{sanitized_name}_info.txt")
            with open(expert_info_file, 'w') as f:
                for key, value in expert_info.items():
                    f.write(f"{key}: {value}\n")

            print(f"Saved expert info for {expert_name}")
            return redirect(url_for('admin_upload'))

    # Dynamically load expert details
    expert_details = []  # Fetch expert details here if needed
    if os.path.exists(app.config['EXPERT_INFO_FOLDER']):
        for info_file in os.listdir(app.config['EXPERT_INFO_FOLDER']):
            with open(os.path.join(app.config['EXPERT_INFO_FOLDER'], info_file), 'r') as f:
                details = {}
                for line in f:
                    key, value = line.strip().split(': ', 1)
                    details[key] = value
                expert_details.append(details)

    return render_template('admin.html', expert_details=expert_details)



# Step 7: Display list of experts
@app.route('/', methods=['GET', 'POST'])
def home():
    search_results = []
    user_input = ""
    if request.method == 'POST':
        search_input = request.form['search_input']
        user_input = search_input  # Store the user's input to display later
        cleaned_input = clean_text(search_input)
        input_embedding = get_text_embedding(cleaned_input)

        if input_embedding is None:
            return "Error generating embedding for search input", 500

        # Dictionary to store similarity scores for all experts
        expert_similarities = []

        # Loop through all PDFs in UPLOAD_FOLDER to calculate similarity
        for pdf_filename in os.listdir(app.config['UPLOAD_FOLDER']):
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
            pdf_text, error = extract_text_from_pdf(pdf_path)
            if error:
                print(f"Skipping {pdf_filename} due to extraction error: {error}")
                continue

            summarized_text = summarize_pdf_text(pdf_text)
            pdf_embedding = get_text_embedding(clean_text(summarized_text))
            if pdf_embedding is None:
                continue

            # Calculate similarity
            similarity = util.pytorch_cos_sim(input_embedding, pdf_embedding).item()

            # Normalize similarity based on word count
            word_count = len(summarized_text.split())
            similarity = adjust_similarity(similarity, word_count, min_words=1200, max_words=15000)

            print(f"Normalized similarity for {pdf_filename} (word count: {word_count}): {similarity}")

            # Collect expert information
            expert_info_file = os.path.join(app.config['EXPERT_INFO_FOLDER'], f"{pdf_filename.replace('.pdf', '')}_info.txt")
            expert_info = {}
            if os.path.exists(expert_info_file):
                with open(expert_info_file, 'r') as f:
                    for line in f.readlines():
                        key, value = line.strip().split(': ', 1)
                        expert_info[key] = value

            # Append similarity and expert details
            expert_similarities.append((pdf_filename, similarity, expert_info))

        # Sort experts by similarity in descending order
        expert_similarities.sort(key=lambda x: x[1], reverse=True)

        # Identify the highest similarity score
        if expert_similarities:
            highest_similarity = expert_similarities[0][1]
            threshold = highest_similarity - 0.03

            # Filter experts within 0.05 range of the highest similarity
            for pdf_filename, similarity, expert_info in expert_similarities:
                if similarity >= threshold:
                    search_results.append({
                        'name': expert_info.get('name', 'Unknown'),
                        'email': expert_info.get('email', 'N/A'),
                        'faculty': expert_info.get('faculty', 'N/A'),
                        'similarity': similarity,
                        'word_count': word_count
                    })

    # Render results in the home.html template along with the user's input
    return render_template('home.html', search_results=search_results, user_input=user_input)


@app.route('/delete/<filename>', methods=['POST'])
def delete_expert(filename):
    # Sanitize filename to prevent directory traversal attacks
    sanitized_filename = secure_filename(filename)

    # Path to the PDF and info files
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], sanitized_filename)
    info_path = os.path.join(app.config['EXPERT_INFO_FOLDER'], f"{sanitized_filename.replace('.pdf', '')}_info.txt")

    # Debugging: Print paths to ensure they are correct
    print(f"Attempting to delete: {pdf_path} and {info_path}")

    # Delete PDF file if it exists
    pdf_deleted = False
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        print(f"Deleted PDF: {pdf_path}")
        pdf_deleted = True

    # Delete info file if it exists
    info_deleted = False
    if os.path.exists(info_path):
        os.remove(info_path)
        print(f"Deleted info file: {info_path}")
        info_deleted = True

    # Check if both files were successfully deleted
    if pdf_deleted or info_deleted:
        flash(f"Successfully deleted: {sanitized_filename}")
    else:
        flash(f"File not found: {sanitized_filename}")

    return redirect(url_for('admin_upload'))




@app.route('/download/<filename>', methods=['GET'])
def download_pdf(filename):
    # Ensure .pdf is appended if not present
    if not filename.endswith('.pdf'):
        filename += ".pdf"
    sanitized_filename = filename.replace('.', '')  # Removes the dot ('.') from the filename
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"Attempting to download file: {file_path}")  # Debugging

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return f"File '{filename}' not found", 404

    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


# Run the Flask app
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['EXPERT_INFO_FOLDER']):
        os.makedirs(app.config['EXPERT_INFO_FOLDER'])
    app.run(debug=True)