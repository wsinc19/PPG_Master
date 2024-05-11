import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, BatchNormalization

# Путь к папке с файлами .csv
folder_path = "PPG-ADData"

# Списки для хранения данных из всех файлов
X = []
y = []

# Загрузка данных из всех файлов в папке
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        # Загрузка данных из файла .csv
        df = pd.read_csv(file_path, header=None)

        # Извлечение данных PPG и ABP
        ppg_data = df.iloc[:, 0].values.reshape(-1, 1)  # Данные PPG
        abp_data = df.iloc[:, 1].values.reshape(-1, 1)  # Данные ABP

        # Добавление данных PPG и ABP в список
        X.append(ppg_data)
        y.append(abp_data)

print("Данные загружены")

# Объединение списков в один массив
X = np.vstack(X)
y = np.vstack(y).ravel()  # Преобразование в одномерный массив

print(X[1], X[2], X[3], X[4])
print(y[1], y[2], y[3], y[4])

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Данные обработаны")

# Создание модели нейронной сети
model = Sequential()

# Добавление слоев
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(units=1))

# Компиляция модели с выбором функции потерь и оптимизатора
model.compile(optimizer='adam', loss='mse')

# Определение коллбэков для ранней остановки и сохранения лучшей модели
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model1.h5', monitor='val_loss', verbose=1, save_best_only=True)

# Обучение модели с использованием валидационного набора данных
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2,
                    callbacks=[early_stopping, checkpoint])

# Оценка модели на тестовых данных
mse = model.evaluate(X_test, y_test)
print("Mean Squared Error:", mse)

# Загрузка лучшей модели
best_model = load_model('best_model1.h5')

# Предсказание на тестовом наборе данных
y_pred = best_model.predict(X_test)

# Оценка производительности модели
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)