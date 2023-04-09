import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import tensorflow as tf
from gensim.parsing.preprocessing import remove_stopwords

# Установить использование памяти наращиванием вместо выделения всей памяти вначале
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Задаем пути до файлов с данными
ham_folder_path = 'D:/Новая папка/ham'  # Папка с неспам-письмами
spam_folder_path = 'D:/Новая папка/spam'  # Папка со спам-письмами

# Функция, которая считывает данные и метки из папок и убирает стоп-слова
def read_files_from_folders(folder_path):
    texts = []
    labels = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            # Убираем стоп-слова
            text = remove_stopwords(text)
            texts.append(text)
            labels.append(0 if folder_path == ham_folder_path else 1)  # 0, если письмо не спам, 1 - если спам
    return texts, labels

ham_texts, ham_labels = read_files_from_folders(ham_folder_path)
spam_texts, spam_labels = read_files_from_folders(spam_folder_path)

# Объединяем данные в один массив
texts = ham_texts + spam_texts
labels = ham_labels + spam_labels

# Инициализируем токенизатор
tokenizer = Tokenizer(num_words=5000, split=' ')
tokenizer.fit_on_texts(texts)

# Преобразуем тексты в числовые последовательности
X = tokenizer.texts_to_sequences(texts)

# Добавляем заполнение для текстов меньшей длины
X = pad_sequences(X)

# Разбиваем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)
# Задаем пути до файлов с данными
ham_folder_path = 'D:/Новая папка/ham'  # Папка с неспам-письмами
spam_folder_path = 'D:/Новая папка/spam'  # Папка со спам-письмами

def predict_message_type(user_input):
    user_input = [user_input]
    user_input = tokenizer.texts_to_sequences(user_input)
    user_input = pad_sequences(user_input, maxlen=X.shape[1])
    prediction = model.predict(np.array(user_input))
    return "spam" if prediction[0][0] > 0.5 else "not spam"

#Проверяем, есть ли файлы с данными обучения и метками
if os.path.exists('X_train.npy') and os.path.exists('X_test.npy') and os.path.exists('y_train.npy') and os.path.exists('y_test.npy'):
  # Загружаем данные обучения и метки
  X_train = np.load('X_train.npy')
  X_test = np.load('X_test.npy')
  y_train = np.load('y_train.npy')
  y_test = np.load('y_test.npy')
  embedding_vector_length = 32
  model = Sequential()
  model.add(Embedding(5000, embedding_vector_length, input_length=X.shape[1]))
  model.add(SpatialDropout1D(0.4))
  model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  user_input = input("Enter your message: ")
  prediction = predict_message_type(user_input)
  print("Your message is", prediction)

else:
    # Ограничиваем оперативную память до 11 ГБ
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
        except RuntimeError as e:
            print(e)
    # Задаем пути до файлов с данными
    ham_folder_path = '/ham'  # Папка с неспам-письмами
    spam_folder_path = '/spam'  # Папка со спам-письмами



    ham_texts, ham_labels = read_files_from_folders(ham_folder_path)
    spam_texts, spam_labels = read_files_from_folders(spam_folder_path)

    # Объединяем данные в один массив
    texts = ham_texts + spam_texts
    labels = ham_labels + spam_labels

    # Инициализируем токенизатор
    tokenizer = Tokenizer(num_words=5000, split=' ')
    tokenizer.fit_on_texts(texts)

    # Преобразуем тексты в числовые последовательности
    X = tokenizer.texts_to_sequences(texts)

    # Добавляем заполнение для текстов меньшей длины
    X = pad_sequences(X)

    # Разбиваем на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

    # Изменяем тип меток на массив numpy
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Описываем архитектуру модели
    embedding_vector_length = 32
    model = Sequential()
    model.add(Embedding(5000, embedding_vector_length, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Выполняем обучение
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=64)

    # Проверяем качество обучения на тестовой выборке
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    #Сохраняем данные обучения и метки в отдельные файлы
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

    # Пример использования функции
    user_input = input("Enter your message: ")
    prediction = predict_message_type(user_input)
    print("Your message is", prediction)