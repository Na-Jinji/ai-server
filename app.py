from flask import Flask, render_template, request, jsonify
from model import model
from konlpy.tag import Kkma

# kkma = Kkma()
app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='AI-SERVER')

@app.route('/userLogin', methods=['POST'])
def userLogin():
    user = request.get_json()   # json 데이터를 받아온다.
    return jsonify(user)    # 받아온 데이터를 다시 전송한다.

@app.route('/environments/<language>')
def environments(language):
    return jsonify({"language": language})

@app.route('/recommender')
def make_recommendation():
    # jpype.attachThreadToJVM()

    if request.method == 'POST':
        landmark = request.get_json()  # json 데이터를 받는다.

        if 'name' in landmark.keys():
            result = model.return_recommendations(landmark['name'])
            return jsonify({"recommended_landmarks": result})
        else:
            return render_template('recommender.html', label="wrong data type")
    else:
        landmark = request.args.to_dict()
        if len(landmark) == 0:
            return 'No parameter'

        if 'name' in landmark.keys():
            result = model.return_recommendations(landmark['name'])
            return jsonify({"recommended_landmarks": result})

if __name__ == '__main__':
    app.run(debug=True)
