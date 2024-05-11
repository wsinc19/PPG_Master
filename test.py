import scipy.io

# Путь к файлу .mat
file_path = "PPG - AD Data/array_45.mat"

# Получение списка переменных в файле .mat
variables = scipy.io.whosmat(file_path)

# Вывод списка переменных
print(variables)