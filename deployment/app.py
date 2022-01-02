from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv("DataAfterMask.csv", encoding="latin-1")
    df.drop(['clickbait', 'topices'], axis=1, inplace=True)
    df.dropna(inplace=True)
    # Features and Labels
    df['label'] = df['popular_topices'].map({'Entertainment': 0, 'Politics': 1, 'Lifestyle': 2, 'Crimes': 3  })
    X = df['headline']
    y = df['label']

    # Extract Feature With CountVectorizer
    cv= CountVectorizer(stop_words='english')
    X = cv.fit_transform(X)
    #X_test = cv.transform(X_test)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #X_test = cv.transform(list(X_test))

    clf  = LogisticRegression()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
	app.run(debug=True)