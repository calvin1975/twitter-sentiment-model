import flask
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Use pickle to load in the pre-trained model.
with open(f'model/Sentiment_LRmodel.pkl', 'rb') as f:
    model = pickle.load(f)

with open(f'model/tf_idf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'POST':
        text1 = flask.request.form.get('text')
        text = [text1]

        vector = TfidfVectorizer(vocabulary = tfidf.vocabulary_)
        text = vector.fit_transform(text)

        print(text)

        pred = model.predict(text)
        return flask.render_template('main.html', original_input= text1, result= pred)
    else:
        return flask.render_template('main.html')

if __name__ == "__main__":
    app.run(debug=True, port=5000) #run app in debug mode on port 5000