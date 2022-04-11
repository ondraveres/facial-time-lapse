from flask import send_from_directory
import time
from io import BytesIO
from main import align_image, invertImage, createGif, saveImagesFromGoogleSearch, removeOthers
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
            # try:
            pathsAndAges.append(
                ('http://halmos.felk.cvut.cz:5000/storage/'+align_image(file.filename)[0], align_image(file.filename)[1]))
            # except:
            #     print('error')
        response = jsonify(pathsAndAges)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


@app.route('/generateGif2API', methods=['GET', 'POST'])
def gif_gen2():
    if request.method == 'POST':
        content = request.get_json()
        paths = []
        ages = []
        for item in content:
            paths.append('../storage/'+item['path'])
            ages.append(item['age'])

        pathToGif = createGif(paths, ages, 2)
        response = make_response(send_file(pathToGif))
        response.headers.add('Access-Control-Allow-Origin', '*')
        os.remove(pathToGif)
        return response


@app.route('/generateGif3API', methods=['GET', 'POST'])
def gif_gen3():
    if request.method == 'POST':
        content = request.get_json()
        paths = []
        ages = []
        for item in content:
            paths.append('../storage/'+item['path'])
            ages.append(item['age'])

        pathToGif = createGif(paths, ages, 3)
        response = make_response(send_file(pathToGif))
        response.headers.add('Access-Control-Allow-Origin', '*')
        os.remove(pathToGif)
        return response


@ app.route('/uploadNameAPI', methods=['GET', 'POST'])
def name_to_gif():
    if request.method == 'POST':
        content = request.get_json()
        name = content['celebrityName']

        pathsAndAges = saveImagesFromGoogleSearch(name, 3)
        # tic = time.time()

        # pathsAndAges = removeOthers(pathsAndAges)
        # toc = time.time()

        #print('removeOthers took {:.4f} seconds.'.format(toc - tic))
        response = jsonify(pathsAndAges)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


@app.route('/storage/<path:path>')
def send_report(path):
    return send_from_directory('../storage', path)


if __name__ == '__main__':
    app.run(host="0.0.0.0")
