import dearpygui.dearpygui as dpg
import serial
import matplotlib.pyplot as plt
import time
import os
from keras.models import load_model
import numpy as np
import csv
from scipy import interpolate

dpg.create_context()

looprun = 0
received_data = []  # Initialize an empty list to store received data
time_stamps = []  # Initialize an empty list to store time stamps


def restart():
    global received_data, time_stamps
    received_data.clear()
    time_stamps.clear()
    dpg.set_value('series_tag', [[], []])


def predict_pressure(ppg, model_path='my_model.h5'):
    # Загрузка сохраненной модели
    model1 = load_model(model_path)

    # Проведение предсказания на модели
    abp_pred = model1.predict(ppg)

    # Вычисление нижней и верхней границ давления
    lower_bound = np.min(abp_pred)  # Минимальное предсказанное значение давления
    upper_bound = np.max(abp_pred)  # Максимальное предсказанное значение давления

    return lower_bound, upper_bound


def ai_callback(sender, app_data):
    global received_data, time_stamps

    lower_bound = 1.0  # Нижний предел диапазона давления
    upper_bound = 2.5  # Верхний предел диапазона давления

    ppg_data = np.array(received_data).reshape(-1, 1)

    # Проверка и интерполяция данных PPG, если необходимо
    if np.min(ppg_data) < lower_bound or np.max(ppg_data) > upper_bound:
        # Создание функции интерполяции на основе текущих данных PPG
        interp_func = interpolate.interp1d(
            np.linspace(0, 1, len(ppg_data)),  # Используем равномерный интервал [0, 1]
            ppg_data.flatten(),  # Преобразуем данные PPG в одномерный массив
            kind='linear',  # Линейная интерполяция
            fill_value='extrapolate'  # Экстраполяция за пределы исходного диапазона
        )

        # Генерация новых данных PPG в нужном диапазоне
        new_ppg_data = interp_func(np.linspace(0, 1, len(ppg_data)))  # Используем новый интервал [0, 1]

        # Приведение данных PPG к диапазону [lower_bound, upper_bound]
        scaled_ppg_data = lower_bound + (upper_bound - lower_bound) * (new_ppg_data - np.min(new_ppg_data)) / (
                np.max(new_ppg_data) - np.min(new_ppg_data))

        # Использование отмасштабированных данных PPG для дальнейшей обработки
        ppg_data = scaled_ppg_data.reshape(-1, 1)

    # Отобразить исходные и обработанные данные PPG на графике
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, 1, len(ppg_data)), ppg_data, label='Processed PPG Data', color='b', linestyle='-')

    # Отметить границы диапазона [lower_bound, upper_bound]
    plt.axhline(y=lower_bound, color='r', linestyle='--', label='Lower Bound')
    plt.axhline(y=upper_bound, color='g', linestyle='--', label='Upper Bound')

    # Настройки графика
    plt.title('Processed PPG Data with Pressure Bounds')
    plt.xlabel('Normalized Time')
    plt.ylabel('PPG Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Здесь вызывайте функцию predict_pressure с обработанными данными PPG
    lower, upper = predict_pressure(ppg_data)

    print("Lower Pressure Bound:", lower)
    print("Upper Pressure Bound:", upper)


def start_ai():
    ai_callback(None, None)


def read_serial_thread(comport, baudrate, folder_path):
    global looprun, received_data, time_stamps  # Доступ к глобальным спискам
    ser = None

    try:
        file_name = "PPG_test.csv"  # Имя файла, который вы хотите создать
        file_path = os.path.join(folder_path, file_name)  # Полный путь к файлу
        print('File Log Ready')
    except Exception as b:
        print("error saving file: " + str(b))

    try:
        ser = serial.Serial(comport, baudrate)
        print('Succesful Connected to Serial Port COM:' + comport + '  Baudrate:' + baudrate)
    except Exception as a:
        print("error open serial port: " + str(a))
        return

    start_time = time.time()  # начало записи

    with open(file_path, "w", newline='') as csvfile:  # Открываем файл в режиме записи
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Time", "Data"])  # Записываем заголовок CSV

        if ser.isOpen():
            print("OK")
            try:
                ser.flushInput()  # flush input buffer, discarding all its contents
                ser.flushOutput()  # flush output buffer, aborting current output
                # and discard all that is in buffer

                while looprun == 1 and time.time() - start_time < 60:  # Record data for one minute
                    line = ser.readline().decode().strip()  # decode bytes to string and strip extra characters
                    current_time = int((time.time() - start_time) * 1000)  # время с начала записи в миллисекундах

                    # Записываем время и данные в файл CSV
                    csv_writer.writerow([current_time, line])
                    print(f"{current_time}, {line}")  # write data to output

                    # Store received data and time stamps
                    received_data.append(float(line))  # Assuming the received data is numerical
                    time_stamps.append(current_time)

                    update_series()

            except Exception as e1:
                print("error" + str(e1))
            finally:
                looprun = 0
                # plot_received_data()
                print("end")

        else:
            print("cannot open serial port ")


def update_series():
    global received_data, time_stamps  # Доступ к глобальным спискам

    dpg.set_value('series_tag', [time_stamps, received_data])
    dpg.set_item_label('series_tag', "PPG")

    #dpg.set_axis_limits_auto("x_axis")
    dpg.fit_axis_data("x_axis")

    #dpg.set_axis_limits_auto("y_axis")
    dpg.fit_axis_data("y_axis")


def plot_received_data():
    global received_data, time_stamps  # Access the global lists
    plt.figure(figsize=(20, 8))
    plt.plot(time_stamps, received_data)
    plt.xlabel('Time')
    plt.ylabel('Received Data')
    plt.title('Received Data over Time')
    plt.grid(True)
    plt.show()


def callback(sender, app_data):
    global selected_folder_path
    selected_folder_path = app_data['file_path_name']


def start():
    global looprun
    looprun = 1
    global selected_folder_path
    comport = dpg.get_value("comport_list")
    baudrate = dpg.get_value("baudrate_list")
    read_serial_thread(comport, baudrate, selected_folder_path)


dpg.add_file_dialog(directory_selector=True, show=False, callback=callback, tag="file_dialog_id", width=500, height=300)

with dpg.font_registry():
    default_font = dpg.add_font("AgfriquercItalic.otf", 20)
    second_font = dpg.add_font("AgfriquercItalic.otf", 13)

with dpg.window(tag="Primary Window"):
    dpg.bind_font(second_font)

    b1 = dpg.add_text("Smart AD", indent=250)
    with dpg.group(horizontal=True):
        with dpg.group(horizontal=False, width=450):
            with dpg.plot(label="PPG", height=300, width=400):
                # optionally create legend
                dpg.add_plot_legend()

                # REQUIRED: create x and y axes
                dpg.add_plot_axis(dpg.mvXAxis, label="time, ms", tag="x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="PPG value", tag="y_axis")

                # series belong to a y axis
                dpg.add_line_series(time_stamps, received_data, label="PPG", parent="y_axis", tag="series_tag")

        with dpg.group(horizontal=False, width=250, indent=500):
            dpg.add_text("Please, select a folder:", indent=45)
            dpg.add_button(label="Directory Selector", callback=lambda: dpg.show_item("file_dialog_id"))
            dpg.add_text("Port")
            comport_list = ["COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "COM10", "COM11", "COM12", "COM13"]
            dpg.add_listbox(label="", items=comport_list, width=200, num_items=2, tag="comport_list")
            dpg.add_text("Baud Rate")
            baudrate_list = ["9600", "38400", "57600", "115200"]
            dpg.add_listbox(label="", items=baudrate_list, width=200, num_items=2, tag="baudrate_list")
            dpg.add_button(label="Start", callback=start)
            dpg.add_button(label="AI", callback=start_ai)
            dpg.add_button(label="Restart", callback=restart)
    dpg.bind_item_font(b1, default_font)

with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (12, 58, 71), category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)

    with dpg.theme_component(dpg.mvInputInt):
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (95, 214, 250), category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)

    with dpg.theme_component(dpg.mvThemeCol_WindowBg):
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (15, 47, 56), category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)

dpg.bind_theme(global_theme)
dpg.create_viewport(title='Custom Title', width=800, height=420)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()