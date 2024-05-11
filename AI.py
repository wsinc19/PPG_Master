import os
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Путь к папке с файлами
folder_path = 'csv_data'

# Список для хранения данных
all_signal_data = []
all_pressure_data = []

# Итерация по файлам в папке
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)

        # Определяем тип данных (ABP или PPG) по имени файла
        if 'PPG_Signal' in filename:
            # Чтение данных сигналов PPG
            data = pd.read_csv(file_path)
            all_signal_data.append(data)

        elif 'ABP_Signal' in filename:
            # Чтение данных давления ABP
            data = pd.read_csv(file_path)
            all_pressure_data.append(data)

# Преобразование данных в массивы numpy
X_train = np.concatenate(all_signal_data, axis=0)
y_train = np.concatenate(all_pressure_data, axis=0)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Определение архитектуры нейронной сети
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(2)  # Выходной слой с двумя нейронами для предсказания верхней и нижней границ давления
])

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Сохранение модели в файл
model.save('my_model.h5')

# Оценка качества модели
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
