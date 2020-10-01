# Text-Classification-Python

This repository covers whole range of text classification problems using different machine learning algorithms.

# Installation
The general installation guide to run different projects is provided here. However if any **error** occurs due to missing libraries, please read the error and install the library according to that information.

```
pip install nltk
pip install tensorflow
pip install Keras
pip install pandas
pip install matplotlib
pip install sklearn
pip install numpy
```
> There are issues regarding the installation of Tensorflow. To check versioning and other aspects, please click this [link](https://github.com/Yunus0or1/Object-Detection-Python/blob/master/README.md) to make a clear understanding of tensoflow installation guide. 

___
> Source code explanations
___

There is a urge necessity to use **Embedding Layer** in neural network to do text classification. To understand why, hit this [Medium](https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12) article. 

## Neural_Network_classification

 - Convolutional Neural Network in action to do text classification.
 - Layers: Embedding&#8594;Conv1D&#8594;MaxPooling1D&#8594;Conv1D&#8594;MaxPooling1D&#8594;LSTM&#8594;Dense
 - ***softmax*** activatation is used in the last layer to do a normalized probability distribution among multiple classes.
 - Data Labels
 
   ```
   1- Traffic Jam
   2- No Traffic Jam
   3- Road Condition
   6- Accident
   7- Fire
   ````
 
## NER-Python

 - An NER system using preposition to extract location from social media posts.
 - Uses NLTK library to get the **Parts of Speech** tags and identify place names on three steps.
 - All the POS tags along with a video tutorial can be found in this [link](https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/).
 - The program analyses the given String and look up different prepositions in order to find a valid location name

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
