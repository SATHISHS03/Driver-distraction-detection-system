from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

#Load the model
test_model = load_model('vgg16_DDD.h5')

activity_map = {'c0': 'Safe driving', 
                'c1': 'Texting - right', 
                'c2': 'Talking on the phone - right', 
                'c3': 'Texting - left', 
                'c4': 'Talking on the phone - left', 
                'c5': 'Operating the radio', 
                'c6': 'Drinking', 
                'c7': 'Reaching behind', 
                'c8': 'Hair and makeup', 
                'c9': 'Talking to passenger'}

# Load the image
img = Image.open('img_3.jpg')

# Convert to RGB format
img_rgb = img.convert('RGB')

# Resize the image
img_resized = img_rgb.resize((224, 224))

# Convert to NumPy array
img_array = np.array(img_resized)

# Reshape to add batch dimension
img_batch = np.expand_dims(img_array, axis=0)


# Make a prediction
prediction = test_model.predict(img_batch)

# Get the predicted class label
predicted_class_index = np.argmax(prediction)
predicted_class_label = activity_map[f'c{predicted_class_index}']
print(predicted_class_label)