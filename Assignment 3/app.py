from flask import Flask, request, render_template, url_for, redirect
import joblib
import score

app = Flask(__name__)

filename = open("C:/Users/Rishika Tibrewal/OneDrive/Desktop/Applied-Machine-Learning/Assignment 2/mlpmodel.joblib",'rb')
mlp =joblib.load(filename)

threshold=0.5


@app.route('/') 
def home():
    return render_template('spam.html')


@app.route('/spam', methods=['POST'])
def spam():
    sent = request.form['sent']
    label,prop=score.score(sent,mlp,threshold)
    lbl="Spam" if label == 1 else "not a spam"
    ans = f"""The sentence "{sent}" is {lbl} with propensity {prop} """
    return render_template('res.html', ans=ans)


if __name__ == '__main__': 
    app.run(debug=True)