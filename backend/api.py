from flask import send_from_directory
import time
from io import BytesIO
from main import align_image, invertImage, createGif, saveImagesFromGoogleSearch, removeOthers, run_pixel_experiment
from flask import Flask, render_template, request, send_file, make_response, jsonify
from flask_cors import CORS
import os
app = Flask(__name__)
CORS(app)


@app.route('/uploadFileAPI', methods=['POST'])
def file_uploader():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("image")
        pathsAndAges = []
        for file in uploaded_files:
            file.save(file.filename)
            try:
                pathsAndAges += align_image(file.filename)
            except Exception as e:
                print(e)
        if(len(pathsAndAges) >= 4):
            pathsAndAges = removeOthers(pathsAndAges)

        response = jsonify(pathsAndAges)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


@ app.route('/uploadNameAPI', methods=['GET', 'POST'])
def name_to_gif():
    if request.method == 'POST':
        content = request.get_json()
        name = content['celebrityName']

        pathsAndAges = saveImagesFromGoogleSearch(name, 3)
        # tic = time.time()

        pathsAndAges = removeOthers(pathsAndAges)
        # toc = time.time()

        #print('removeOthers took {:.4f} seconds.'.format(toc - tic))
        response = jsonify(pathsAndAges)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


@app.route('/generateGifAPI', methods=['GET', 'POST'])
def gif_gen():
    if request.method == 'POST':
        content = request.get_json()
        paths = []
        ages = []
        items = content['items']
        max_opacity = float(int(content['opacity']))/100
        encoder = content['encoder']
        frames = int(content['frames'])
        output_size = int(content['size'])

        for item in items:
            paths.append('../storage/'+item['path'])
            ages.append(item['age'])

        pathToGif = None
        if(encoder == 'pixel'):
            pathToGif = run_pixel_experiment(paths, frames)
        else:
            pathToGif = createGif(paths, ages, encoder,
                                  frames, frames/2, max_opacity, output_size)
        response = make_response(send_file(pathToGif))
        response.headers.add('Access-Control-Allow-Origin', '*')
        os.remove(pathToGif)
        return response


@app.route('/storage/<path:path>')
def send_report(path):
    return send_from_directory('../storage', path)


if __name__ == '__main__':
    app.run(host="0.0.0.0")
