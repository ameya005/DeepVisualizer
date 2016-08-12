#Visualization tools for CNN

##First layer filters:

Usage: python CNN_visualizer.py path to model

##Visualizing activations for filters in a layer:

keras_translate.py

This code translates the model into keras and uses gradient ascent to maximise the activation from each filter in the layer. Thus, using back propogation and a random noise image as input, the program displays the shapes and normalized colors used by the filter.

Usage: python keras\_translate.py path to model layer number starting from_1
