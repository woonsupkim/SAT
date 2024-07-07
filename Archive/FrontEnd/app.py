from flask import Flask, request, jsonify, send_from_directory, render_template
import pandas as pd

app = Flask(__name__)

# Load the CSV file
df = pd.read_csv('SAT_Question_Bank_Results.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/next-question', methods=['POST'])
def next_question():
    data = request.json
    current_question_index = data['current_question_index']
    user_answers = data['user_answers']
    answer_times = data['answer_times']

    # Determine the next question index
    next_question_index = (current_question_index + 1) % len(df)  # Example logic

    return jsonify({'next_question_index': next_question_index})

@app.route('/get-question', methods=['GET'])
def get_question():
    index = int(request.args.get('index', 0))
    if 0 <= index < len(df):
        question_data = df.iloc[index].to_dict()
        return jsonify(question_data)
    return jsonify({'error': 'Index out of range'}), 404

@app.route('/save-results', methods=['POST'])
def save_results():
    data = request.json
    user_answers = data['user_answers']
    answer_times = data['answer_times']

    # Create a DataFrame
    result_df = pd.DataFrame({
        'Question': range(1, len(user_answers) + 1),
        'Answer': user_answers,
        'Time': answer_times
    })

    # Optionally, save the DataFrame to a CSV file or any other storage
    result_df.to_csv('student_answers.csv', index=False)

    return jsonify(result_df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
