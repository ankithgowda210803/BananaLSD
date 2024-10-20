from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow import keras
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the model
model = load_model('Trained_Model.keras')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    if request.method == 'POST':
     
        img_file = request.files['image']
        
       
        img_path = 'temp_image.jpg'
        img_file.save(img_path)
      
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        
        
        class_index = np.argmax(prediction)
        predicted_label = chr(class_index + 65)  
        
       
        os.remove(img_path)
        
        return render_template('result.html', predicted_label=predicted_label)


if __name__ == '__main__':
    app.run(debug=True)
