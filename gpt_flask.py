import firebase_admin
from firebase_admin import credentials,firestore
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
from PIL import Image
from decimal import Decimal, ROUND_HALF_UP
#---------------------------- initialize app -------------------------
app = Flask(__name__)
# --------------------------- initilaize firebaase --------------------
#cred = credentials.Certificate("E:/Desktop/Grad-Project/test model/flutter-test1-omarelheny-firebase-adminsdk-wq0rx-3cc12d5bba.json")
# cred=credentials.Certificate("D:/Extras/Codefiles/Flutterfiles/college_project/api/graduationproject-fd501-firebase-adminsdk-4rg5f-865efffb59.json")
#firebase_admin.initialize_app(cred)
#db = firestore.client()

# Load your AI model
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
# Define a function to preprocess the image
def preprocess_image(image):
    # Load the image
    img = image

    # Resize the image to match the input size expected by the model
    img = img.resize((224, 224))

    # Convert the image to an array
    img_array = np.array(img)

    # Normalize the image data
    img_array = img_array / 255.0

    # Expand the dimensions to match the input shape expected by the model
    img_array = np.expand_dims(img_array, axis=0)

    return img_array



# Define the number you want to round
# num = Decimal(1.2225)

# Round to the 3rd decimal place using ROUND_HALF_UP
# rounded_num = num.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)

# print(rounded_num)  # Output: 1.223

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        # if 'file' not in request.data:
        #     return jsonify({'error': 'No file part'})
        file = request.files['file']
        # if file.filename == '':
        #     return jsonify({'error': 'No selected file'})
        if file:

            image = Image.open(file)

            # need test in this part
            # image = Image.open(image).convert("RGB")


            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            # res= db.collection('images').document()
            # res.set({
            #     'prediction': class_name,
            #     "Confidence Score:": confidence_score
            # })


            # Return the prediction as a response

            return jsonify({'prediction': class_name},{"Confidence Score": str(confidence_score)})


if __name__ == '__main__':
    app.run(debug=True)
