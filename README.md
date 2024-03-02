# Product Categorization using NLP and LSTM

## Introduction

This project aims to categorize products using Natural Language Processing (NLP) techniques, specifically leveraging LSTM layers in a neural network architecture. The goal is to classify products into predefined categories based on their textual descriptions. Tensorflow library have been used for the NLP and model training.

## Data Processing

The dataset consists of product descriptions and their corresponding categories. The data preprocessing steps include:

- Data inspection on how the dataset is being presented. The data set currently have labels and text description although a row of data had NaN value so it was removed
- Isolating features and labels from the dataset.
- Performing label encoding on the category column to convert categorical labels into numerical values.
- Tokenizing and padding the text data to prepare it for input into the LSTM model.

## Tokenization and Embedding

For tokenization, we utilized the `Tokenizer` class from the Keras library. The tokenizer converts text data into sequences of integers, which are then padded to ensure uniform length sequences. Additionally, we used word embeddings to represent words in a continuous vector space for better classification and model performance later when training it.

## Model Training

The LSTM model was trained using the following parameters:

- **Epochs**: 30
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy

The architecture of the LSTM model includes multiple LSTM layers followed by dense layers with softmax activation for multiclass classification.
![model_architecture](https://github.com/FlameCerberus/Products-Categorization-using-LSTM/assets/96816249/eea7c949-6df3-4a4f-87ea-52c38da01ec9)

- **Training Loss and Validation Loss**: The loss calculated during the training and validation phases, respectively.
  ![Epoch_Loss_Tensorboard](https://github.com/FlameCerberus/Products-Categorization-using-LSTM/assets/96816249/76ed3f31-abbb-439b-9394-6025364644c7)

- **Training Accuracy and Validation Accuracy**: The accuracy of the model on the training and validation data, respectively.
  ![Epoch_Accuracy_Tensorboard](https://github.com/FlameCerberus/Products-Categorization-using-LSTM/assets/96816249/aba62e62-abb7-4942-a935-965ad9355c4c)

The training shows that there are no overfitting or underfitting occuring in the tensorboard training images as the validation and training accuracy do not curve down.

## Model Evaluation

The model was evaluated using the following metrics:

- **F1 Score and Accuracy**: A combination of the F1 score, which provides a balance between precision and recall, and the accuracy score, which measures the proportion of correct predictions among the total number of predictions.
![F1,Accuracy-Scores](https://github.com/FlameCerberus/Products-Categorization-using-LSTM/assets/96816249/b0fcf19b-21cf-4232-bf41-9808a18aa5bb)

The F1 and Accuracy score shows a great promise on the model's ability to predict the product's category based on its description given to the model as F1 score is very high.

- **Confusion Matrix**: A table showing the counts of true positive, true negative, false positive, and false negative predictions.
![Confusion_Matrix](https://github.com/FlameCerberus/Products-Categorization-using-LSTM/assets/96816249/77477ee9-62a0-47f7-b8eb-07333c8b97ae)

Most of the predicted values are shown in the confusion matrix have been accurate as a lot of the prediction are true positives. This strengthen the information from the F1 and Accuracy score

## Live Test Results

We conducted live testing on a sample of unseen data to assess the model's performance in real-world scenarios by deploying the model using streamlit showing it can predict product's categories by giving it the description
![image](https://github.com/FlameCerberus/Products-Categorization-using-LSTM/assets/96816249/faa98658-088a-4d2f-b60d-4ae57c136148)

The input given was:
The ROG Strix G15 is a gaming laptop from ASUS. It has a 15.6-inch screen, 165 Hz refresh rate, and WQHD resolution. It also has a 512 GB NVMe PCIe M.2 SSD, USB Type-A and Type-C ports, HDMI, Wi-Fi 6, Gigabit LAN, and Bluetooth 5.0.
- AMD Ryzen™ 9 6900HX CPU
- 4800MHz DDR5 RAM
- ROG Intelligent CoolingTM with liquid metal and four fan outlets
- Smart Amp speakers with Dolby Atmos
- NVIDIA GeForce RTXTM 3070 Ti Laptop GPU with 150W max TGP
- PCIe 4.0
- 84 curved Arc Flow FansTM
- Large touchpad
- Backlit keyboard
- Combo audio jack


The model predicted the description to be electronics which is a true statement as what was given is a laptop specifications which is an electronic category of an item.


## Deployment Using Streamlit

To deploy the model for real-time predictions, the project utilized Streamlit, a popular Python library for building interactive web applications. The deployment process involves running the model_deployment.py using streamlit run.

## Conclusion

In conclusion, the project achieved its objective on building a model to predict the category of a product based on its description given. the project demonstrates the effectiveness of using NLP techniques, specifically LSTM neural networks, for product categorization tasks. By leveraging deep learning models, we can achieve accurate and efficient classification of products based on their textual descriptions.

## References

[1] [Dataset used from Kaggle](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification)

[2] [Tokenizer in Python](https://www.analyticsvidhya.com/blog/2020/05/what-is-tokenization-nlp/)

[3] [TensorFlow Guide on NLP](https://www.tensorflow.org/tutorials/text)
