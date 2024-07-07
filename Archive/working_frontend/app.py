from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/next-question', methods=['POST'])
def next_question():
    data = request.json
    current_question_index = data['current_question_index']
    user_answers = data['user_answers']
    answer_times = data['answer_times']

    # Run your algorithm to get the next question index
    next_question_index = (current_question_index + 1) % len(user_answers)  # Example logic

    return jsonify({'next_question_index': next_question_index})

@app.route('/save-results', methods=['POST'])
def save_results():
    data = request.json
    user_answers = data['user_answers']
    answer_times = data['answer_times']

    # Create a DataFrame
    df = pd.DataFrame({
        'Question': range(1, len(user_answers) + 1),
        'Answer': user_answers,
        'Time': answer_times
    })

    # Optionally, save the DataFrame to a CSV file or any other storage
    df.to_csv('student_answers.csv
