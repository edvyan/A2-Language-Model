# Dash LSTM Text Generation Application

## Overview
This Dash application allows users to input a text prompt, which is then fed to a pre-trained LSTM (Long Short-Term Memory) language model. The model generates a continuation of the input text, demonstrating a simple form of natural language processing and text prediction. 

## Components of the Web Application
The Dash application is composed of following components: 
1. Dash Framework   
2. Text Input  
3. Output Display  

## LSTM Language Model

### 1. Model Definition
The `LSTMLanguageModel` class defines an LSTM-based neural network for text generation. It includes embedding layers, LSTM layers, dropout layers, and a fully connected layer to output the probability distribution over the vocabulary.

### 2. Model Initialization and Loading
The LSTM model is initialized with specific dimensions and parameters like vocabulary size, embedding dimensions, hidden dimensions, number of layers, and dropout rate. The pre-trained model's state dictionary is loaded from a file, enabling the model to generate text based on its prior training.

### 3. Tokenizer and Vocabulary
The tokenizer, in this case, a basic English tokenizer from `torchtext`, splits input text into tokens (words). The vocabulary, loaded from a file, maps these tokens to numerical indices, which the LSTM model can process.

## Text Generation Process

### 1. Callback Function for Text Generation
The `generate_continuation` function serves as a callback in Dash. It's triggered when the user inputs text and generates the text continuation. The function processes the input text, feeds it to the LSTM model, and then formats the model's output into a human-readable string.

### 2. Generating Continuation
The input prompt is tokenized and converted into indices. The LSTM model, in a state of no gradient calculation to optimize performance (`torch.no_grad()`), generates a sequence of words based on the input. A softmax layer, combined with a temperature parameter, is used to generate probabilities for the next word in the sequence. The sequence generation continues until the maximum sequence length is reached or an end-of-sentence token is generated.

### 3. Output Formatting
The generated indices are converted back to tokens (words), which are then joined into a string and displayed in the web application.

## Running the Application
To run the application, please open a terminal in the app folder and type:

python app.py