import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('general_image_classifier.h5')
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def predict(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))
    return {class_names[i]: float(preds[0][i]) for i in range(len(class_names))}

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="General Image Classifier",
    description="Upload an image to see the predicted class."
).launch()
