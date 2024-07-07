document.addEventListener('DOMContentLoaded', () => {
    let currentQuestionIndex = 0;
    let timer;
    let timeElapsed = 0;

    const questionImage = document.getElementById('question-image');
    const answerForm = document.getElementById('answer-form');
    const feedback = document.getElementById('feedback');
    const nextButton = document.getElementById('next-question');
    const showExplanationButton = document.getElementById('show-explanation');
    const timeDisplay = document.getElementById('time-elapsed');
    const explanationSection = document.getElementById('explanation-section');
    const explanationImage = document.getElementById('explanation-image');

    let userAnswers = JSON.parse(localStorage.getItem('userAnswers')) || [];
    let answerTimes = JSON.parse(localStorage.getItem('answerTimes')) || [];

    function loadQuestion(index) {
        fetch(`http://localhost:5000/get-question?index=${index}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            questionImage.src = `data:image/png;base64,${data['Question Image2']}`;
            feedback.textContent = '';
            answerForm.answer.value = userAnswers[index] || '';
            explanationSection.style.display = 'none';

            // Reset and start the timer
            clearInterval(timer);
            timeElapsed = 0;
            timeDisplay.textContent = timeElapsed;
            timer = setInterval(() => {
                timeElapsed++;
                timeDisplay.textContent = timeElapsed;
            }, 1000);
        })
        .catch(error => {
            feedback.textContent = 'Error loading the question.';
            console.error('Error:', error);
        });
    }

    answerForm.addEventListener('submit', (event) => {
        event.preventDefault();
        const userAnswer = event.target.answer.value.trim();

        // Stop the timer and save the time elapsed
        clearInterval(timer);
        answerTimes[currentQuestionIndex] = timeElapsed;

        userAnswers[currentQuestionIndex] = userAnswer;
        localStorage.setItem('userAnswers', JSON.stringify(userAnswers));
        localStorage.setItem('answerTimes', JSON.stringify(answerTimes));

        if (userAnswer) {
            feedback.textContent = `Your answer has been submitted! Time taken: ${timeElapsed} seconds`;

            // Send the results to the server
            sendResultsToServer();
        } else {
            feedback.textContent = 'Please enter an answer.';
        }
    });

    nextButton.addEventListener('click', () => {
        fetchNextQuestion();
    });

    showExplanationButton.addEventListener('click', () => {
        displayExplanation();
    });

    function fetchNextQuestion() {
        fetch('http://localhost:5000/next-question', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                current_question_index: currentQuestionIndex,
                user_answers: userAnswers,
                answer_times: answerTimes
            }),
        })
        .then(response => response.json())
        .then(data => {
            currentQuestionIndex = data.next_question_index;
            loadQuestion(currentQuestionIndex);
        })
        .catch(error => {
            feedback.textContent = 'Error fetching the next question.';
            console.error('Error:', error);
        });
    }

    function sendResultsToServer() {
        fetch('http://localhost:5000/save-results', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_answers: userAnswers,
                answer_times: answerTimes
            }),
        })
        .then(response => response.json())
        .then(data => {
            console.log('Results saved successfully:', data);
        })
        .catch(error => {
            feedback.textContent = 'Error saving the results.';
            console.error('Error:', error);
        });
    }

    function displayExplanation() {
        fetch(`http://localhost:5000/get-question?index=${currentQuestionIndex}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            explanationImage.src = `data:image/png;base64,${data['Rationale Image2']}`;
            explanationSection.style.display = 'block';
        })
        .catch(error => {
            feedback.textContent = 'Error loading the explanation.';
            console.error('Error:', error);
        });
    }

    loadQuestion(currentQuestionIndex);
});
