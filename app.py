import os
import pymysql
import cv2
import numpy as np
import pytesseract
from flask import Flask, request, jsonify, render_template
from ai71 import AI71
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from werkzeug.utils import secure_filename
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


app = Flask(__name__, template_folder='template', static_folder='static')

# Firebase setup
cred = credentials.Certificate('/Users/rosh/Downloads/falcon-50f06-firebase-adminsdk-no87w-d32c464aa6.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# AI71 API Key
AI71_API_KEY = "api71-api-20725a9d-46d6-4baf-9e26-abfca35ab242"
client = AI71(AI71_API_KEY)

# Function to get data from the database
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/student')
def student():
    return render_template('student.html')

@app.route('/teacher')
def teacher():
    return render_template('teacher.html')

@app.route('/generate-timetable', methods=['POST'])
def generate_timetable():
    data = request.json
    hours_per_day = data.get('hours_per_day')
    days_per_week = data.get('days_per_week')
    semester_end_date = data.get('semester_end_date')
    subjects = data.get('subjects', [])

    # Input validation
    if not hours_per_day or not days_per_week or not semester_end_date or not subjects:
        return jsonify({"error": "Missing required parameters"}), 400

    try:
        # Simple invocation of the AI71 API
        response = client.chat.completions.create(
            model="tiiuae/falcon-180B-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Create a timetable starting from Monday based on the following inputs:\n"
                                            f"- Number of hours per day: {hours_per_day}\n"
                                            f"- Number of days per week: {days_per_week}\n"
                                            f"- Semester end date: {semester_end_date}\n"
                                            f"- Subjects: {', '.join(subjects)}\n"}
            ]
        )

        # Access the response content correctly
        timetable = response.choices[0].message.content if response.choices and response.choices[0].message else "No timetable generated."
        return jsonify({"timetable": timetable})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# New functions for question paper generation
def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def generate_questions_from_text(text, no_of_questions, marks_per_part, no_parts):
    ai71 = AI71(AI71_API_KEY)
    messages = [
        {"role": "system", "content": "You are a teaching assistant"},
        {"role": "user",
         "content": f"Give your own {no_of_questions} questions under each part for {no_parts} parts with {marks_per_part} marks for each part. Note that all questions must be from the topics of {text}"}
    ]

    questions = []
    for chunk in ai71.chat.completions.create(
            model="tiiuae/falcon-180b-chat",
            messages=messages,
            stream=True,
    ):
        if chunk.choices[0].delta.content:
            questions.append(chunk.choices[0].delta.content)

    return "".join(questions)

@app.route('/generate-paper', methods=['GET', 'POST'])
def generate_paper():
    if request.method == 'POST':
        no_of_questions = int(request.form['no_of_questions'])
        total_marks = int(request.form['total_marks'])
        no_of_parts = int(request.form['no_of_parts'])
        marks_per_part = int(request.form['marks_per_part'])
        test_duration = request.form['test_duration']
        pdf_file = request.files['pdf_file']

        if pdf_file:
            # Secure the file name and save the file to the upload folder
            filename = secure_filename(pdf_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            pdf_file.save(file_path)

            # Extract text from the curriculum PDF
            curriculum_text = extract_text_from_pdf(file_path)

            # Generate questions
            questions = generate_questions_from_text(curriculum_text, no_of_questions, marks_per_part, no_of_parts)

            # Optionally, remove the saved file after use
            os.remove(file_path)

            return render_template('paper_gen.html', questions=questions)

    return render_template('paper_gen.html')


def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img)
    return text


def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    final_text = ""
    for image in images:
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        text = pytesseract.image_to_string(image_cv)
        final_text += text
    return final_text


def evaluate(question, answer, max_marks):
    prompt = f"""Questions: {question}
    Answer: {answer}.


    Evaluate above questions one by one(if there are multiple) by provided answers and assign marks out of {max_marks}. No need overall score. Note that as maximum mark increases, the size of the answer must be large enough to get good marks."""

    messages = [
        {"role": "system", "content": "You are an answer evaluator"},
        {"role": "user", "content": prompt}
    ]

    response_content = ""
    for chunk in client.chat.completions.create(
            model="tiiuae/falcon-180b-chat",
            messages=messages,
            stream=True
    ):
        if chunk.choices[0].delta.content:
            response_content += chunk.choices[0].delta.content

    return response_content


@app.route('/eval', methods=['GET', 'POST'])
def eval():
    if request.method == 'POST':
        input_type = request.form['input_type']
        question_text = ""
        answer_text = ""
        max_marks = request.form['max_marks']

        if input_type == 'file':
            question_file = request.files['question_file']
            answer_file = request.files['answer_file']

            if question_file and answer_file:
                question_path = os.path.join(app.config['UPLOAD_FOLDER'], question_file.filename)
                answer_path = os.path.join(app.config['UPLOAD_FOLDER'], answer_file.filename)

                question_file.save(question_path)
                answer_file.save(answer_path)

                if question_path.endswith('.pdf'):
                    question_text = extract_text_from_pdf(question_path)
                else:
                    question_text = extract_text_from_image(question_path)

                if answer_path.endswith('.pdf'):
                    answer_text = extract_text_from_pdf(answer_path)
                else:
                    answer_text = extract_text_from_image(answer_path)

        elif input_type == 'text':
            question_text = request.form['question_text']
            answer_text = request.form['answer_text']

        evaluation_result = evaluate(question_text, answer_text, max_marks)
        print(f"Question Text: {question_text}")  # Debugging line
        print(f"Answer Text: {answer_text}")  # Debugging line
        print(f"Evaluation Result: {evaluation_result}")  # Debugging line

        return render_template('result.html', result=evaluation_result)

    return render_template('eval.html')

@app.route('/get_students')
def get_students():
    # Retrieve data from Firestore
    students_ref = db.collection('students')
    docs = students_ref.stream()
    students = [doc.to_dict() for doc in docs]
    return jsonify(students)


@app.route('/generate_report', methods=['POST'])
def generate_report():
    student_data = request.json
    report = generate_student_report(
        student_data['name'],
        student_data['age'],
        student_data['cgpa'],
        student_data['course_pursuing'],
        student_data['assigned_test_score'],
        student_data['ai_test_score'],
        student_data['interests'],
        student_data['difficulty_in'],
        student_data['courses_taken']
    )
    return jsonify({'report': report})


def generate_student_report(name, age, cgpa, course, assigned_test, ai_test, interests, difficulty, courses_taken):
    prompt = f"""
    Name: {name}
    Age: {age}
    CGPA: {cgpa}
    Course: {course}
    Assigned Test Score: {assigned_test}
    AI generated Test Score: {ai_test}
    Interests: {interests}
    Difficulty in: {difficulty}
    Courses Taken: {courses_taken}
    Use the above student data to generate a neat personalized report and suggested teaching methods."""

    client = AI71(AI71_API_KEY)

    response = client.chat.completions.create(
        model="tiiuae/falcon-180B-chat",
        messages=[
            {"role": "system", "content": "You are a student report generator."},
            {"role": "user", "content": prompt}
        ]
    )

    report = response.choices[0].message.content if response.choices and response.choices[
        0].message else "No report generated."

    return report


if __name__ == '__main__':
    app.run(debug=True)
