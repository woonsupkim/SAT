import streamlit as st
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import os

def load_data(subject):
    file_path = 'Streamlit/SAT_math.csv' if subject == 'math' else 'Streamlit/SAT_reading.csv'
    
    if not os.path.isfile(file_path):
        st.error(f"File {file_path} not found. Please ensure the file is uploaded.")
        st.stop()

    df = pd.read_csv(file_path)

    le_skill = LabelEncoder()
    le_difficulty = LabelEncoder()
    le_answer = LabelEncoder()

    df['Skill'] = le_skill.fit_transform(df['Skill'])
    df['Question Difficulty'] = le_difficulty.fit_transform(df['Question Difficulty'])

    if 'time_spent' not in df.columns:
        df['time_spent'] = 0

    questions = [f'data:image/png;base64,{img}' for img in df["Question Image2"]]
    explanations = [f'data:image/png;base64,{img}' for img in df["Rationale Image2"]]
    df['Correct Answer Encoded'] = le_answer.fit_transform(df['Correct Answer'])

    return df, questions, explanations, le_answer

def get_image_html(base64_str):
    return f'<img src="{base64_str}" style="max-width: 100%; height: auto; border: 1px solid #ddd; padding: 10px; background-color: #fff;">'

def suggest_next_question(df, user_answers, elapsed_time):
    if 'model' in st.session_state and 'scaler' in st.session_state:
        potential_questions = df[['Skill', 'Question Difficulty', 'time_spent']].copy()
        potential_questions['time_spent'] = np.mean(elapsed_time) if elapsed_time else 0
        potential_questions_scaled = st.session_state.scaler.transform(potential_questions)

        probabilities = st.session_state.model.predict_proba(potential_questions_scaled)
        if probabilities.shape[1] > 1:
            next_question_index = np.argmin(np.abs(probabilities[:, 1] - 0.5))
        else:
            next_question_index = np.random.randint(0, len(df))
    else:
        next_question_index = np.random.randint(0, len(df))
    
    return next_question_index

def main():
    st.set_page_config(page_title="SAT Study Platform", page_icon="🎓", layout="centered")

    st.markdown("""
    <style>
        .main {
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
        }
        .stButton button {
            background-color: #3498db;
            color: white;
            padding: 15px 30px;
            margin: 10px;
            border: none;
            cursor: pointer;
            border-radius: 10px;
            font-size: 18px;
            display: flex;
            align-items: center;
        }
        .stButton button:hover {
            background-color: #2980b9;
        }
        .stProgress > div > div > div > div {
            background-color: #3498db;
        }
        .stButton button img {
            margin-right: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    if st.session_state.page == 'home':
        st.header("Welcome to the SAT Study Platform")
        st.markdown("This SAT study platform presents questions with images, tracks time spent, records answers, and provides explanations. It uses a machine learning model to suggest the next question based on user performance, adapting to individual learning needs.")
        
        st.markdown("### Instructions")
        st.markdown("""
        1. Select the subject you want to study by clicking on the "Math" or "Reading and Writing" button.
        2. Answer the questions presented.
        3. View explanations after answering by clicking on "Show Explanation".
        4. Track your progress with the progress bar at the top.
        5. Provide feedback by clicking on the "User Feedback" button.
        """)
        
        st.markdown("### Select the subject you want to study:")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Math 🧮"):
                st.session_state.page = 'math'
                reset_session_state('math')
                st.experimental_rerun()
                
        with col2:
            if st.button("Reading and Writing 📖"):
                st.session_state.page = 'reading_writing'
                reset_session_state('reading_writing')
                st.experimental_rerun()

        if st.button("User Feedback 📝"):
            st.session_state.page = 'feedback'
            st.experimental_rerun()

    elif st.session_state.page == 'math':
        study_subject('math')
    elif st.session_state.page == 'reading_writing':
        study_subject('reading_writing')
    elif st.session_state.page == 'feedback':
        user_feedback()

def reset_session_state(subject):
    df, questions, explanations, le_answer = load_data(subject)
    st.session_state.df = df
    st.session_state.questions = questions
    st.session_state.explanations = explanations
    st.session_state.le_answer = le_answer
    st.session_state.current_question_index = np.random.randint(0, len(df))
    st.session_state.user_answers = [''] * len(questions)
    st.session_state.answer_times = [0] * len(questions)
    st.session_state.start_time = time.time()
    st.session_state.elapsed_time = 0
    st.session_state.timer_running = True
    st.session_state.show_explanation = False
    st.session_state.questions_answered = 0

def study_subject(subject):
    st.markdown("""
    <style>
        .main {
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
        }
        .stButton button {
            background-color: #3498db;
            color: white;
            padding: 15px 30px;
            margin: 10px;
            border: none;
            cursor: pointer;
            border-radius: 10px;
            font-size: 18px;
            display: flex;
            align-items: center;
        }
        .stButton button:hover {
            background-color: #2980b9;
        }
    </style>
    """, unsafe_allow_html=True)
    
    if subject == 'math':
        st.title("Math Section")
    else:
        st.title("Reading and Writing Section")

    if st.button("Back to Home 🏠"):
        st.session_state.page = 'home'
        st.experimental_rerun()

    if 'current_question_index' not in st.session_state:
        reset_session_state(subject)

    df = st.session_state.df
    questions = st.session_state.questions
    explanations = st.session_state.explanations
    le_answer = st.session_state.le_answer

    current_question_index = st.session_state.current_question_index
    user_answers = st.session_state.user_answers
    answer_times = st.session_state.answer_times
    start_time = st.session_state.start_time
    elapsed_time = st.session_state.elapsed_time
    timer_running = st.session_state.timer_running
    show_explanation = st.session_state.show_explanation

    # Progress bar
    total_questions = len(df)
    questions_answered = st.session_state.questions_answered
    progress = questions_answered / total_questions
    st.progress(progress)

    question_html = get_image_html(questions[current_question_index])
    explanation_html = get_image_html(explanations[current_question_index])

    st.markdown(question_html, unsafe_allow_html=True)

    with st.form(key='answer_form'):
        answer = st.text_input("Your Answer:", value="", key="answer_input")
        submit = st.form_submit_button("Submit 📨")

    show_explanation_btn = st.button("Show Explanation 📜", key="show_explanation_btn")

    if submit:
        user_answers[current_question_index] = answer
        answer_times[current_question_index] = int(elapsed_time)
        st.session_state.user_answers = user_answers
        st.session_state.answer_times = answer_times
        st.session_state.questions_answered += 1
        st.success(f"Your answer has been submitted! Time taken: {answer_times[current_question_index]} seconds")
        st.session_state.timer_running = False

        correct_answer = df.loc[current_question_index, 'Correct Answer Encoded']
        encoded_answer = le_answer.transform([answer.strip().lower()])[0] if answer.strip().lower() in le_answer.classes_ else -1
        is_correct = 1 if encoded_answer == correct_answer else -1

        skill = df.loc[current_question_index, 'Skill']
        difficulty = df.loc[current_question_index, 'Question Difficulty']
        new_data = pd.DataFrame({'Skill': [skill], 'Question Difficulty': [difficulty], 'time_spent': [elapsed_time]})
        new_target = pd.Series([is_correct])

        if 'X_train' in st.session_state and 'y_train' in st.session_state:
            st.session_state.X_train = pd.concat([st.session_state.X_train, new_data], ignore_index=True)
            st.session_state.y_train = pd.concat([st.session_state.y_train, new_target], ignore_index=True)
        else:
            st.session_state.X_train = new_data
            st.session_state.y_train = new_target

        if st.session_state.questions_answered % 5 == 0:
            if len(st.session_state.y_train.unique()) > 1:
                st.session_state.scaler = StandardScaler().fit(st.session_state.X_train)
                st.session_state.X_train = st.session_state.scaler.transform(st.session_state.X_train)
                st.session_state.model = RandomForestClassifier(n_estimators=100)
                st.session_state.model.fit(st.session_state.X_train, st.session_state.y_train)
            else:
                st.warning("Not enough variability in the target variable to retrain the model.")

        results_df = pd.DataFrame({
            'Question Index': list(range(len(user_answers))),
            'User Answer': user_answers,
            'Time Taken (seconds)': answer_times
        })
        results_df.to_csv('Streamlit/user_results.csv', index=False)

        st.session_state.current_question_index = suggest_next_question(df, user_answers, answer_times)
        st.session_state.start_time = time.time()
        st.session_state.elapsed_time = 0
        st.session_state.timer_running = True
        st.session_state.show_explanation = False
        st.experimental_rerun()

    if show_explanation_btn:
        st.session_state.show_explanation = True

    if show_explanation:
        st.markdown(explanation_html, unsafe_allow_html=True)

    if timer_running:
        elapsed_time = int(time.time() - start_time)
        st.session_state.elapsed_time = elapsed_time

    timer_placeholder = st.empty()
    timer_placeholder.write(f"Time: {elapsed_time} seconds")

    if timer_running:
        time.sleep(1)
        st.experimental_rerun()

def user_feedback():
    st.header("User Feedback")
    feedback = st.text_area("Please provide your feedback below:")
    if st.button("Submit Feedback 📤"):
        with open("feedback.txt", "a") as f:
            f.write(feedback + "\n")
        st.success("Thank you for your feedback!")
    if st.button("Back to Home 🏠"):
        st.session_state.page = 'home'
        st.experimental_rerun()

if __name__ == "__main__":
    main()
