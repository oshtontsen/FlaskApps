from flask import Flask, flash, request, url_for, redirect, render_template
import pickle
import os
import numpy as np
from werkzeug.utils import secure_filename
from flask import send_from_directory
import numpy as np
from PIL import Image




# model=pickle.load(open('model.pkl','rb'))

# The types of uploads should be limited to
# prevent HTML or php injection attacks.
UPLOAD_FOLDER = 'C:/Users/OSHX1/Documents/Projects/WebDev/Digitizer/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload New File</title>
    <h1>Upload New File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Converts the image into a numpy array, gray scales the image, and resizes the image
    im = np.array(Image.open('uploads/%s' % filename).convert('L').resize((28,28)))
    print(im)
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

# @app.route('/')
# def index():
#     return render_template("digitizer.html")


# @app.route('/predict',methods=['POST','GET'])
# def predict():
#     int_features=[int(x) for x in request.form.values()
#     final=[np.array(int_features)]
#     print(int_features)
#     print(final)
#     prediction=model.predict_proba(final)
#     output='{0:.{1}f}'.format(prediction[0][1], 2)
#
#     if output>str(0.5):
#         return render_template('forest_fire.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
#     else:
#         return render_template('forest_fire.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")
#
# app.config["IMAGE_UPLOADS"] = "/mnt/c/wsl/projects/pythonise/tutorials/flask_series/app/app/static/img/uploads"
# app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]



if __name__ == '__main__':
    app.run(debug=True)
