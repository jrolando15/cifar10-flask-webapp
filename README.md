# cifar10-flask-webapp

## Project Description
This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

With this model, we obtained a test accuracy of 0.80 on the CIFAR-10 dataset.

Additionally, this project includes a Flask web application that allows users to upload images and get predictions from the trained model.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Model Saving](#model-saving)
- [Web Application](#web-application)
- [Prediction](#prediction)
- [License](#license)

## Installation
To run this project, you need to have Python installed along with the following libraries:
- TensorFlow
- NumPy
- Flask
- Matplotlib

You can install the required libraries using the following command:
```bash
pip install tensorflow numpy flask matplotlib
```

## Usage
1) Clone the repository:
```bash
git clone https://github.com/jrolando15/cifar10-flask-webapp.git
cd Image-Classification-with-CIFAR-10
```

2) Run the Jupyter notebook
```bash
jupyter notebook
```

3) Start Flask web application:
```bash
python main.py
```
Open your web browser and go to http://127.0.0.1:5000 to interact with the web app.

## Project Structure
```bash
Image-Classification-with-CIFAR-10/
├── app.py                             # Flask web application
├── image_classification_cifar10.ipynb # Jupyter notebook with the code
├── model/                             # Folder containing the trained model
│   └── cifar10_model.h5               # Saved model (after training)
├── templates/                         # HTML templates for the web app
│   ├── index.html
│   ├── upload_form.html
│   └── prediction.html
├── static/                            # Static files (CSS, JS, images)
│   └── style.css
├── README.md                          # Project README file
└── requirements.txt                   # List of dependencies
```

## Data Processing
The CIFAR-10 dataset is loaded using the cifar10.load_data() function from TensorFlow's Keras datasets. The images are normalized, and the labels are one-hot encoded.
```bash
from tensorflow.keras.datasets import cifar10
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
num_classes = 10
y_train = np.eye(num_classes)[y_train.reshape(-1)]
y_test = np.eye(num_classes)[y_test.reshape(-1)]
```

## Model Architecture
The CNN model is defined using TensorFlow's Keras Sequential API. The model consists of the following layers:
Convolutional Layer (32 filters, 3x3 kernel)
Convolutional Layer (32 filters, 3x3 kernel)
Max Pooling Layer (2x2 pool size)
Dropout Layer (0.25)
Convolutional Layer (64 filters, 3x3 kernel)
Convolutional Layer (64 filters, 3x3 kernel)
Max Pooling Layer (2x2 pool size)
Dropout Layer (0.25)
Flatten Layer
Dense Layer (512 units, ReLU activation)
Dropout Layer (0.5)
Dense Layer (10 units, Softmax activation)

```bash
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

```
## Model Training
The model is compiled with the Adam optimizer and categorical cross-entropy loss function. The model is trained for 30 epochs with a batch size of 128.

```bash
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 30
batch_size = 128

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)
```

## Model Evaluation
The trained model is evaluated on the test set, and the test loss and accuracy are printed.
```bash
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores)
print('Test accuracy:', scores[1])
```
## Model Saving
model.save('cifar10_model.h5')

## Web application
main.py
```bash
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
```

index.html
```bash
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CIFAR-10 Image Prediction</title>
    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <!-- Optional JavaScript -->
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.2/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</head>
<body>
    <div class="container mt-5">
        <h1 class="mb-4">Upload an image for CIFAR-10 prediction</h1>
        <form action="/predict" method="post" enctype="multipart/form-data" class="mb-3">
            <div class="form-group">
                <input type="file" name="image" class="form-control-file" required>
            </div>
            <button type="submit" class="btn btn-primary">Predict</button>
        </form>
        {% if prediction %}
            <div class="alert alert-success" role="alert">
                <h4 class="alert-heading">Prediction: {{ prediction }}</h4>
                <img src="{{ img_path }}" alt="Uploaded Image" class="img-fluid">
            </div>
        {% endif %}
    </div>
</body>
</html>
```
## License
This README file provides a comprehensive overview of your project, including its description, installation instructions, usage, project structure, data processing, model architecture, training, evaluation, web application, and prediction. It also includes sections for contributing and licensing, which are important for open-source projects.

