from flask import Flask, render_template, request, jsonify
from model import model

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html',title='AI-SERVER')

@app.route('/<pagename>')
def hello(pagename):
    return pagename

@app.route('/animal')
def animal():
    return render_template('animal.html')

@app.route('/dog')
def dog():
    return render_template('dog.html')

@app.route('/userLogin', methods=['POST'])
def userLogin():
    user = request.get_json()#json 데이터를 받아온다.
    return jsonify(user)#받아온 데이터를 다시 전송한다.

@app.route('/environments/<language>')
def environments(language):
    return jsonify({"language": language})

@app.route('/recommender', methods=['POST'])
def make_recommendation():
    if request.method == 'POST':
        landmark = request.get_json()#json 데이터를 받는다.
        if 'name' in landmark.keys():
            result = model.test(landmark['name'])
            return jsonify({"recommended_landmarks": result})
        else:
            return render_template('recommender.html',label="wrong data type")


if __name__ == '__main__':
    app.run(debug=True)
