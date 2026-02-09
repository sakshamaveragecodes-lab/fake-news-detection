from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    data = vectorizer.transform([news])
    prediction = model.predict(data)[0]
    result = "Fake News ❌" if prediction == 1 else "Real News ✅"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
