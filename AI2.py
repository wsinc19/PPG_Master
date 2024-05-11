import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten


# Функция для вычисления инженерных признаков
def calculate_engineered_features(data, window_size):
    # Вычисление статистических показателей для каждого окна данных PPG
    features = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        mean = np.mean(window)
        median = np.median(window)
        std_dev = np.std(window)
        features.append([mean, median, std_dev])
    return np.array(features).reshape(-1, window_size, 1)



# Путь к папке с файлами .csv
folder_path = "PPG-ADData"
window_size = 100  # Размер временного окна для вычисления признаков

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
        ppg_data = df.iloc[:, 0].values  # Данные PPG
        abp_data = df.iloc[:, 1].values  # Данные ABP

        # Вычисление инженерных признаков для данных PPG
        ppg_features = calculate_engineered_features(ppg_data, window_size)

        for i in range(len(ppg_features)):
            X.append(ppg_features[i])
            y.append(abp_data[i + window_size - 1])  # ABP соответствует последнему значению в каждом окне PPG

print("Данные загружены")

# Преобразование списков в массивы numpy
X = np.array(X)
y = np.array(y)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование данных для совместимости с моделью
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print("Данные обработаны")

# Создание модели нейронной сети
model = Sequential()
# Добавление сверточного слоя с 1D ядром размером 3 и 32 фильтрами
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(window_size, 1)))
# Добавление слоя субдискретизации для уменьшения размерности данных
model.add(MaxPooling1D(pool_size=2))
# Преобразование двумерного выхода сверточного слоя в одномерный массив
model.add(Flatten())
# Добавление полносвязного слоя с 50 нейронами
model.add(Dense(units=50, activation='relu'))
# Добавление выходного полносвязного слоя с одним нейроном (для регрессии)
model.add(Dense(units=1))

# Компиляция модели с выбором функции потерь и оптимизатора
model.compile(optimizer='adam', loss='mse')

# Определение коллбэков для ранней остановки и сохранения лучшей модели
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model2.h5', monitor='val_loss', verbose=1, save_best_only=True)

# Обучение модели с использованием валидационного набора данных
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2,
                    callbacks=[early_stopping, checkpoint])

# Оценка модели на тестовых данных
mse = model.evaluate(X_test, y_test)
print("Mean Squared Error:", mse)

# Загрузка лучшей модели
best_model = load_model('best_model2.h5')

# Предсказание на тестовом наборе данных
y_pred = best_model.predict(X_test)

# Оценка производительности модели
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
