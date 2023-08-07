

from flask import Flask, render_template, send_from_directory, url_for
from flask_uploads import UploadSet, IMAGES,configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import tensorflow as tf
from predict import predict

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

file_id = '1-4kY_iSsDf6vA_S68lOL9dmYyXltV1CY'
file = drive.CreateFile({'id': file_id})
file.GetContentFile('network.h5')

network = tf.keras.models.load_model('network.h5')

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'asldfkjlj'
app.config['UPLOADED_PHOTOS_DEST'] = 'pictures'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)


class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Only images can be uploaded.'),
            FileRequired('File field should not be empty')
        ]
    )
    submit = SubmitField('Upload')

@app.route('/pictures/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

@app.route('/', methods=['GET','POST'])
def upload_image():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)
        result = predict(file_url.lstrip('/'))
    else:
        file_url = None
        result = None
    return render_template('index.html', form=form, file_url=file_url, result = result)


if __name__ == '__main__':
    app.run(debug=True)
