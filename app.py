from flask import Flask, request, request, render_template
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)
model = load_model('model/cifar10_model.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    if img_file:
        img_path = 'static/' + img_file.filename
        img_file.save(img_path)
        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        return render_template('index.html', prediction=predicted_class, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
