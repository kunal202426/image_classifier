import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


model = tf.keras.models.load_model('general_image_classifier.h5')


class_names = list(model.class_names) if hasattr(model, 'class_names') else [
    'buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'
]


img_path = 'test_img1.jpg' 


img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)


predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

print(f"This image is predicted to be '{class_names[predicted_class]}' with {confidence:.2f}% confidence.")
