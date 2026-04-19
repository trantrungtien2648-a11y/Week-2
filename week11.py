import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import nltk
import string
import warnings
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
warnings.filterwarnings('ignore')
# Load the dataset
data = pd.read_csv('Dataset---Hate-Speech-Detection-using-Deep-Learning.csv')
# Display the first few rows of the dataset
print(data.head())
print(data.shape)
print(data.info())

plt.pie(
    data['class'].value_counts(), labels=data['class'].value_counts().index, autopct='%1.1f%%', startangle=90
)
plt.title('Distribution Classes')
plt.show()

class_0 = data[data['class'] == 0]
class_1 = data[data['class'] == 1].sample(3500, random_state=42)
class_2 = data[data['class'] == 2]

balanced_data = pd.concat([class_0, class_0, class_0, class_1, class_2], axis=0)

plt.pie(
    balanced_data['class'].value_counts().values,
    labels=balanced_data['class'].value_counts().index,
    autopct='%1.1f%%')

plt.title('Balanced Class Distribution')
plt.show()

data['tweet'] = data['tweet'].str.lower()
punctuations_list = string.punctuation
def remove_punctuation(text):
    term = str.maketrans('', '', punctuations_list)
    return text.translate(term)

data['tweet'] = data['tweet'].apply(lambda x: remove_punctuation(x))
data.head()

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

balanced_data['tweet'] = balanced_data['tweet'].apply(preprocess_text)
balanced_data.head()

def plot_wordcloud(data, typ):
    corpus = " ".join(data['tweet'])
    wc = WordCloud(max_words=100, width=800, height=400, collocations=False).generate(corpus)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for Class {typ} Class", fontsize=15)
    plt.show()

plot_wordcloud(balanced_data[balanced_data['class'] == 2], typ="Neutral")
plot_wordcloud(balanced_data[balanced_data['class'] == 1], typ="Offensive")
plot_wordcloud(balanced_data[balanced_data['class'] == 0], typ="Hate Speech")

features = balanced_data['tweet']
target = balanced_data['class']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

Y_train = pd.get_dummies(pd.Series(y_train, name='class'))
Y_test = pd.get_dummies(pd.Series(y_test, name='class'))

max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

model = keras.models.Sequential([
    layers.Embedding(input_dim=max_words, output_dim=32, input_length=max_len),
    layers.Bidirectional(layers.LSTM(16)),
    layers.Dense(512, activation='relu', kernel_regularizer='l1'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')
])

model.build(input_shape=(None, max_len))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
rlr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=0)
history = model.fit(
    X_train_padded, Y_train, validation_data=(X_test_padded, Y_test), epochs=50, batch_size=32, callbacks=[es, rlr]
)
history_df = pd.DataFrame(history.history)  
history_df[['loss', 'val_loss']].plot(title="loss")
history_df[['accuracy', 'val_accuracy']].plot(title="accuracy")
plt.show()

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test_padded, Y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
