import streamlit as st
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Load the dataset
df = pd.read_csv('SAT_Question_Bank_Results.csv')

# Preprocess the dataset
le_skill = LabelEncoder()
le_difficulty = LabelEncoder()
le_answer = LabelEncoder()

df['Skill'] = le_skill.fit_transform(df['Skill'])
df['Question Difficulty'] = le_difficulty.fit_transform(df['Question Difficulty'])

# Ensure 'time_spent' feature is initialized
if 'time_spent' not in df.columns:
    df['time_spent'] = 0

# Initialize questions and explanations lists from the entire series
questions = [f'data:image/png;base64,{img}' for img in df["Question Image2"]]
explanations = [f'data:image/png;base64,{img}' for img in df["Rationale Image2"]]

# Encode the correct answers
df['Correct Answer Encoded'] = le_answer.fit_transform(df['Correct Answer'])

def get_image_html(base64_str):
    return f'<img src="{base64_str}" style="max-width: 100%; height: auto; border: 1px solid #ddd; padding: 10px; background-color: #fff;">'

def suggest_next_question(user_answers, elapsed_time):
    if 'model' in st.session_state and 'scaler' in st.session_state:
        # Prepare a DataFrame of potential next questions
        potential_questions = df[['Skill', 'Question Difficulty', 'time_spent']].copy()
        potential_questions['time_spent'] = np.mean(elapsed_time) if elapsed_time else 0
        potential_questions_scaled = st.session_state.scaler.transform(potential_questions)

        # Predict the probability of correct answers
        probabilities = st.session_state.model.predict_proba(potential_questions_scaled)
        if probabilities.shape[1] > 1:
            next_question_index = np.argmin(np.abs(probabilities[:, 1] - 0.5))
        else:
            next_question_index = np.random.randint(0, len(df))
    else:
        next_question_index = np.random.randint(0, len(df))
    
    return next_question_index

def main():
    st.title("SAT Study Platform")

    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = np.random.randint(0, len(df))
        st.session_state.user_answers = [''] * len(questions)
        st.session_state.answer_times = [0] * len(questions)
        st.session_state.start_time = time.time()
        st.session_state.elapsed_time = 0
        st.session_state.timer_running = True
        st.session_state.show_explanation = False
        st.session_state.questions_answered = 0

    current_question_index = st.session_state.current_question_index
    user_answers = st.session_state.user_answers
    answer_times = st.session_state.answer_times
    start_time = st.session_state.start_time
    elapsed_time = st.session_state.elapsed_time
    timer_running = st.session_state.timer_running
    show_explanation = st.session_state.show_explanation

    question_html = get_image_html(questions[current_question_index])
    explanation_html = get_image_html(explanations[current_question_index])

    st.markdown(question_html, unsafe_allow_html=True)

    with st.form(key='answer_form'):
        answer = st.text_input("Your Answer:", value="", key="answer_input")
        submit = st.form_submit_button("Submit")

    show_explanation_btn = st.button("Show Explanation", key="show_explanation_btn")

    if submit:
        user_answers[current_question_index] = answer
        answer_times[current_question_index] = int(elapsed_time)
        st.session_state.user_answers = user_answers
        st.session_state.answer_times = answer_times
        st.session_state.questions_answered += 1
        st.success(f"Your answer has been submitted! Time taken: {answer_times[current_question_index]} seconds")
        st.session_state.timer_running = False

        # Check if the submitted answer is correct
        correct_answer = df.loc[current_question_index, 'Correct Answer Encoded']
        encoded_answer = le_answer.transform([answer.strip().lower()])[0] if answer.strip().lower() in le_answer.classes_ else -1
        is_correct = 1 if encoded_answer == correct_answer else -1

        # Update training data
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

        # Check for variability in the target variable before retraining the model
        if st.session_state.questions_answered % 5 == 0:
            if len(st.session_state.y_train.unique()) > 1:
                st.session_state.scaler = StandardScaler().fit(st.session_state.X_train)
                st.session_state.X_train = st.session_state.scaler.transform(st.session_state.X_train)
                st.session_state.model = RandomForestClassifier(n_estimators=100)
                st.session_state.model.fit(st.session_state.X_train, st.session_state.y_train)
            else:
                st.warning("Not enough variability in the target variable to retrain the model.")

        # Save results to a CSV file
        results_df = pd.DataFrame({
            'Question Index': list(range(len(user_answers))),
            'User Answer': user_answers,
            'Time Taken (seconds)': answer_times
        })
        results_df.to_csv('user_results.csv', index=False)

        # Automatically go to the next question
        st.session_state.current_question_index = suggest_next_question(user_answers, answer_times)
        st.session_state.start_time = time.time()
        st.session_state.elapsed_time = 0
        st.session_state.timer_running = True
        st.session_state.show_explanation = False
        st.rerun()

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
        st.rerun()

if __name__ == "__main__":
    main()
