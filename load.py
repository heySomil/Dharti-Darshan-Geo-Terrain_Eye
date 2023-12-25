import numpy as np
import tensorflow as tf

def init(): 
    
    # load model json
    json_file = open('models/model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    # load model
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    
    #load weights into new model
    loaded_model.load_weights("models/model.h5")
    
    print("Loaded Model from disk")

    # compile model 
    loaded_model.compile(
        optimizer = 'adam',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = ['accuracy']
    )
    return loaded_model