from io import BytesIO
from main import align_image, invertImage, createGif, saveImagesFromGoogleSearch
from flask import Flask, render_template, request, send_file, make_response, jsonify
from flask_cors import CORS
import os
app = Flask(__name__)
CORS(app)


@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/uploadFileAPI', methods=['GET', 'POST'])
def file_uploader():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("image")
        pathsAndAges = []
        for file in uploaded_files:
            file.save(file.filename)
            try:
                pathsAndAges.append(align_image(file.filename))
            except:
                print('error')
            os.remove(file.filename)

        # pathToGif = createGif(paths)
        # response = make_response(send_file(pathToGif))
        response = jsonify(pathsAndAges)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


@app.route('/uploadNameAPI', methods=['GET', 'POST'])
def name_to_gif():
    if request.method == 'POST':
        print(request.is_json)
        content = request.get_json()
        name = content['celebrityName']

        pathsAndAges = saveImagesFromGoogleSearch(name, 10)

        response = jsonify(pathsAndAges)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


if __name__ == '__main__':
    app.run(host="0.0.0.0")
