from flask import Flask, render_template, request, session, redirect
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = "secretkey"

model = pickle.load(open("model.pkl", "rb"))
feature_names = model.feature_names_in_

@app.route("/")
def start():
    session["node"] = 0
    return redirect("/question")

@app.route("/question", methods=["GET", "POST"])
def question():
    node = session.get("node", 0)

    # If leaf node
    if model.tree_.feature[node] == -2:
        prediction = model.classes_[np.argmax(model.tree_.value[node])]
        return render_template("result.html", disease=prediction)

    feature_index = model.tree_.feature[node]
    symptom = feature_names[feature_index]

    if request.method == "POST":
        answer = request.form["answer"]

        if answer == "yes":
            node = model.tree_.children_right[node]
        else:
            node = model.tree_.children_left[node]

        session["node"] = int(node)
        return redirect("/question")

    return render_template("question.html", symptom=symptom)

if __name__ == "__main__":
    app.run(debug=True)