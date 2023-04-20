import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

# Load the saved model from file
with tf.device('/cpu:0'):
    model = keras.models.load_model('resnet_50.hdf5')

def save_weights_to_npy():
    # Create an empty state dictionary
    state_dict = {}

    
    # Iterate through the layers of the model and add the weights to the state dict
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        layer_name = layer.name.replace('/', '.')
        for j, weight in enumerate(layer.weights):
            if j == 0:
                weight_name = 'weight'
            elif j == 1:
                weight_name = 'bias'
            else:
                weight_name = weight.name.split('/')[1].split(':')[0].replace('moving','running').replace('variance','var')
            if layer_name not in state_dict:
                state_dict[layer_name] = {}
            
            if len(weights[j].shape) == 4:
                state_dict[layer_name].update({weight_name : weights[j].transpose((3,2,0,1))})
            elif len(weights[j].shape) == 2:
                state_dict[layer_name].update({weight_name : weights[j].transpose()})
            else:
                state_dict[layer_name].update({weight_name : weights[j]})

    # Save the state dict to a file
    np.save('resnet_50.npy',state_dict)

def do_inference():

    # Define a new model with the same inputs and some intermediate layers as outputs
    layer_outputs = [layer.output for layer in model.layers[:-3]]  # Choose some intermediate layers
    intermediate_model = keras.models.Model(inputs=model.inputs, outputs=layer_outputs)

    # Define the shape of the input data
    input_shape = (3, 224, 224)

    # Create an input data array of ones
    input_data = np.ones(input_shape)

    # Reshape the input data for use with Keras
    input_data = np.expand_dims(input_data, axis=0)

    # Do inference on the input data
    output = intermediate_model.predict(input_data)

    print(output)

save_weights_to_npy()
