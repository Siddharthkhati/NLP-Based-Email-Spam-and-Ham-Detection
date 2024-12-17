from flask import Flask, render_template, request
from spamHamClassifier import train_model, predict_text

app = Flask(__name__)
clf, vectorizer = train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        prediction = predict_text(clf, vectorizer, text)
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
