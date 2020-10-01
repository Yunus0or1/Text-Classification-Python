# Text-Classification-Python

This repository covers whole range of text classification problems using different machine learning algorithms.

# 1. Installation & Requirements:
The general installation guide to run different projects is provided here. However if any **error** occurs due to missing libraries, please read the error and install the library according to that information.

<p> Clone the tool if you have git installed. </p>
<b> <ul> Git Installation Guide: </b>
  <li>Windows - https://git-scm.com/download/win </li>
  <li>Linux - https://git-scm.com/download/linux </li>
  </ul>
Then run these command in the Command Prompt or Terminal.

```
git clone https://github.com/Yunus0or1/Text-Classification-Python.git
cd Text-Classification-Python
```
<p> <b>        OR </b> </p>
<p> Download from the link: https://github.com/Yunus0or1/Text-Classification-Python/archive/master.zip <p>
Then, run these command in the Command Prompt or Terminal.

```
cd Text-Classification-Python
```

```
pip install -r requirements.txt
```
> There are issues regarding the installation of Tensorflow. To check versioning and other aspects, please click this [link](https://github.com/Yunus0or1/Object-Detection-Python/blob/master/README.md) to make a clear understanding of tensoflow installation guide. 

___
> Source code explanations
___

There is a urge necessity to use **Embedding Layer** in neural network to do text classification. To understand why, hit this [Medium](https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12) article. 

## Conv-Text-Classification

 - Convolutional Neural Network in action to do text classification.
 - Layers: Embedding&#8594;Conv1D&#8594;MaxPooling1D&#8594;Conv1D&#8594;MaxPooling1D&#8594;LSTM&#8594;Dense
 - ***softmax*** activatation is used to do a normalized probability distribution among multiple classes.

# 2. Usage:
<p> i. Run the python file from the directories. </p>

<p> Or type the command in terminal/command prompt: </p>

```
python _filename_
```
