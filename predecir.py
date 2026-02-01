import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

longitud, altura = 150, 150
ruta_modelo = './modelo/modelo_completo.keras' 

cnn = load_model(ruta_modelo)

def predict(file):
    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    
    array = cnn.predict(x, verbose=0)
    result = array[0]
    answer = np.argmax(result)
    
    clases = ['catedral', 'generalife', 'palacio_de_carlos_v', 'patio_de_los_leones', 'plaza_de_bib-rambla']
    
    if answer < len(clases):
        print(f"Predicción: {clases[answer]}")
    
    return answer

predict('./Data/validacion/dataset_patio_de_los_leones_granada/img_00016.jpg')