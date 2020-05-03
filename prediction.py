import numpy as np
import pickle
from PIL import Image

def make_prediction(im):
    model = pickle.load(open('model.pkl','rb'))
    # Convert the numpy array into a 4D input for the model
    im = im.reshape(-1, 28, 28, 1)
    prediction = model.predict(im)

    return np.argmax(prediction, axis=1)
