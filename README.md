#Visualization tools for CNN

##First layer filters:

Usage: python CNN_visualizer.py <path_to_model>

##Visualizing activations for filters in a layer:

keras_translate.py

This code translates the model into keras and uses gradient ascent to maximise the activation from each filter in the layer. Thus, using back propogation and a random noise image as input, the program displays the shapes and normalized colors used by the filter.

Usage: python keras_translate.py <path_to_model> <layer_number_starting_from_1>
