from flask import Flask, request, render_template
import pickle

# Load trained model
with open("model/spam_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form["email"]
    prediction = model.predict([email_text])[0]
    
    if prediction == 1:
        result = "This mail is a SPAM mail "
    else:
        result = "This mail is NOT a spam mail "
    
    return render_template("index.html", prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
