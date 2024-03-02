# %%
#1. Setup - Import Packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import sklearn,os,pickle
import datetime


# %%
#2. Data Loading
colnames=['Category', 'Text'] 
csv_path = os.path.join(os.getcwd(), 'Dataset/ecommerceDataset.csv')
df = pd.read_csv(csv_path, names=colnames, header=None)
df.info()

# %%
#3. Data Inspection and Data Cleaning
#(A) Data Inspection
print("Shape of the Data: ", df.shape)
print("Data Description: \n", df.describe().transpose())
print("Example Data: ", df.head(1))
df.isnull().sum()

# %%
#(B) Data Cleaning
df.dropna(inplace=True)
df.isnull().sum() 

# %%
#4. Data Preprocessing
#(A) Isolate the features and labels
features = df['Text'].values
labels = df['Category'].values
#(B) Perform label encoding on category column
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# %%
# Checking which encoded label corresponds to its label number
encoded_classes = label_encoder.classes_

# Print the encoded classes
for label, encoded_class in enumerate(encoded_classes):
    print(f"Encoded class {encoded_class} corresponds to label {label}")

# %%
#5. Perform train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,labels_encoded,train_size = 0.8, random_state = 42)

# %%
#6. Start with tokenization
# Define hyperparameters
vocab_size = 5000
oov_token = "<OOV>"
max_length = 200
embedding_dim = 64

# Define the Tokenizer object
tokenizer = keras.preprocessing.text.Tokenizer(
    num_words = vocab_size,
    oov_token = oov_token,
    split=" "
)

tokenizer.fit_on_texts(X_train)

# %%
# Inspection on the Tokenizer
word_index = tokenizer.word_index
print(word_index)

# %%
# Use the tokenizer to transform text to tokens
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)
print(X_train[0])
print(X_test_tokens[0])

# %%
#7. Perform padding and truncating
X_train_padded = keras.preprocessing.sequence.pad_sequences(
    X_train_tokens,
    maxlen=max_length,
    padding = 'post',
    truncating = 'post'
)

X_test_padded = keras.preprocessing.sequence.pad_sequences(
    X_test_tokens,
    maxlen=max_length,
    padding = 'post',
    truncating = 'post'
)
print(X_train_padded.shape)

# %%
# Create the function for the decoding
reversed_word_index = dict([(value,key) for (key,value) in word_index.items()])

def decode_tokens(tokens):
    return " ".join([reversed_word_index.get(i,"?") for i in tokens])

print(X_train[3])
print("------------------------")
print(decode_tokens(X_train_padded[3]))

# %%
#8. Model Development
model = keras.Sequential()
#(A) Create the Embedding Layer to perform token embedding
model.add(keras.layers.Embedding(vocab_size,embedding_dim))
#(B) Proceed to build the RNN as the subsequent layers
model.add(keras.layers.Bidirectional(keras.layers.LSTM(48,return_sequences = False)))
model.add(keras.layers.Dense(len(np.unique(labels)),activation = 'softmax'))
model.summary()

# %%
#9. Compile the model
opt = keras.optimizers.Adam(learning_rate = 0.00001)
model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])

#10. Model Training
early_stopping = keras.callbacks.EarlyStopping(patience = 2)
logpath = os.path.join('tensorboard_log', datetime.datetime.now().strftime('%Y%m%d - %H%M%S'))
tb = keras.callbacks.TensorBoard(logpath)
max_epoch = 30
history = model.fit(X_train_padded, y_train,validation_data =(X_test_padded,y_test),epochs = max_epoch, callbacks = [early_stopping, tb])

# %%
#Training vs Val Loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# %%
#(B) Accuracy Graph
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training Accuracy','Validation Accuracy'])

# %%
#11. Saving token and labels
#(A) Label encoder
label_encoder_save_path = "label_encoder.pkl"
with open(label_encoder_save_path, "wb") as f:
    pickle.dump(label_encoder,f)

#(B) Tokenizer
tokenizer_save_path = "tokenizer.pkl"
with open(tokenizer_save_path, "wb") as f:
    pickle.dump(tokenizer,f)

# %%
# Saving Tokens and Model in a specific format
import json

save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

#12. Saving Token as JSON format
tokenizer_json = tokenizer.to_json()
with open(os.path.join(save_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
    json.dump(tokenizer_json, f)
#13. Saving Model as .h5 format
model.save(os.path.join(save_dir, "model.h5"))

# %%
#(C) Keras Model
model_save_path = "nlp_model"
keras.models.save_model(model,model_save_path)

# %%
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# Make predictions
y_pred_prob = model.predict(X_test_padded)

# Convert probabilities to class labels
y_pred = np.argmax(y_pred_prob, axis=1)

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
acc_score = accuracy_score(y_test, y_pred)
print("F1 Score:", f1)
print("Accuracy Score:", acc_score)

# %%
# Evaluation using confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Encoded class Books corresponds to label 0
# Encoded class Clothing & Accessories corresponds to label 1
# Encoded class Electronics corresponds to label 2
# Encoded class Household corresponds to label 3

# Creating Confusion Matrix for further evaluation of the model using test dataset
def conf_func(y_test, y_pred, figsize=(10, 8), font_scale=1.2, annot_kws_size=16):
    class_names = [0, 1, 2, 3]  # ['Books', 'Clothing & Accessories', 'Electronics ', 'Household']
    tick_marks_y = [0.5, 1.5, 2.5, 3.5]
    tick_marks_x = [0.5, 1.5, 2.5, 3.5]
    confusion_matrix_df = confusion_matrix(y_test, y_pred)
    confusion_matrix_df = pd.DataFrame(confusion_matrix_df, range(4), range(4))
    plt.figure(figsize=figsize)
    sns.set(font_scale=font_scale)  # label size
    plt.title("Confusion Matrix")
    sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": annot_kws_size}, fmt='d')  # font size
    plt.yticks(tick_marks_y, class_names, rotation='vertical')
    plt.xticks(tick_marks_x, class_names, rotation='horizontal')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    plt.show()

conf_func(y_test, y_pred, figsize=(10, 8), font_scale=1.2, annot_kws_size=16)

# %%
from keras.utils import plot_model

# Generate and save model architecture plot as .png
plot_model(model, to_file='model_architecture.png', show_shapes=True)


