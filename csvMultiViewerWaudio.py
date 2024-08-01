import sys
import pandas as pd
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from pyqtgraph.Qt import QtGui, QtCore
import pygame
import glob
from pydub import AudioSegment
from pydub.utils import mediainfo

# トリラテレーション関数
def trilaterate(distances):
    beacon_positions = np.array([
        [0, 0],      # ビーコンA
        [0, 10],     # ビーコンB
        [10, 10],    # ビーコンC
        [10, 0]      # ビーコンD
    ])
    dA, dB, dC, dD = distances
    A = np.array([
        [2 * (beacon_positions[1, 0] - beacon_positions[0, 0]), 2 * (beacon_positions[1, 1] - beacon_positions[0, 1])],
        [2 * (beacon_positions[2, 0] - beacon_positions[1, 0]), 2 * (beacon_positions[2, 1] - beacon_positions[1, 1])],
        [2 * (beacon_positions[3, 0] - beacon_positions[2, 0]), 2 * (beacon_positions[3, 1] - beacon_positions[2, 1])]
    ])
    B = np.array([
        (dA**2 - dB**2 + beacon_positions[1, 0]**2 - beacon_positions[0, 0]**2 + beacon_positions[1, 1]**2 - beacon_positions[0, 1]**2) / 2,
        (dB**2 - dC**2 + beacon_positions[2, 0]**2 - beacon_positions[1, 0]**2 + beacon_positions[2, 1]**2 - beacon_positions[1, 1]**2) / 2,
        (dC**2 - dD**2 + beacon_positions[3, 0]**2 - beacon_positions[2, 0]**2 + beacon_positions[3, 1]**2 - beacon_positions[2, 1]**2) / 2
    ])
    try:
        pos = np.linalg.lstsq(A, B, rcond=None)[0]
        x, y = pos
        if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
            raise ValueError("計算結果が無効です")
    except np.linalg.LinAlgError as e:
        print("線形代数エラー:", e)
        return np.nan, np.nan
    except ValueError as e:
        print("値エラー:", e)
        return np.nan, np.nan
    return x, y

# PyQtGraphアプリケーションの初期化
app = QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(show=True, title="Dance Visualization")
win.resize(1000, 600)
win.setWindowTitle('Dance Visualization')

# より綺麗なプロットのためのアンチエイリアスを有効化
pg.setConfigOptions(antialias=True)

# プロットの追加
plot1 = win.addPlot(title="Position Plot")
plot1.setXRange(-1, 11)
plot1.setYRange(-1, 11)

plot2 = win.addPlot(title="Mean Dance Value")
plot2.setYRange(0, 100)

# 位置の散布図を作成
scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None))

# 平均ダンス値の折れ線グラフを作成
curve = plot2.plot(pen='y')
marker = plot2.plot([0], [0], pen=None, symbol='o', symbolBrush='r')

# 時間選択用のスライダーを追加
slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
slider.setRange(0, 0)
proxy = QtWidgets.QGraphicsProxyWidget()
proxy.setWidget(slider)
win.nextRow()
win.addItem(proxy)

# 再生ボタンの追加
play_button = QtWidgets.QPushButton("Play")
play_button_proxy = QtWidgets.QGraphicsProxyWidget()
play_button_proxy.setWidget(play_button)
win.nextRow()
win.addItem(play_button_proxy)

# フォルダ選択ボタンの追加
folder_button = QtWidgets.QPushButton("Select Folder")
folder_button_proxy = QtWidgets.QGraphicsProxyWidget()
folder_button_proxy.setWidget(folder_button)
win.nextRow()
win.addItem(folder_button_proxy)

# 音声ファイル選択ボタンの追加
sound_button = QtWidgets.QPushButton("Select Sound")
sound_button_proxy = QtWidgets.QGraphicsProxyWidget()
sound_button_proxy.setWidget(sound_button)
win.nextRow()
win.addItem(sound_button_proxy)

# チェックボックスリストを作成
checkbox_layout = QtWidgets.QVBoxLayout()
checkboxes = []
checkbox_widget = QtWidgets.QWidget()
checkbox_widget.setLayout(checkbox_layout)
checkbox_proxy = QtWidgets.QGraphicsProxyWidget()
checkbox_proxy.setWidget(checkbox_widget)
win.nextRow()
win.addItem(checkbox_proxy)

# グローバル変数
playing = False
user_interaction = False
dataframes = []
file_names = []
mean_dance = []

def get_color(dance_value):
    # ブルーからレッドへのグラデーション
    blue = (0, 0, 255)
    red = (255, 0, 0)
    
    # ダンス値を0から100の範囲に正規化し、グラデーションの割合を計算
    ratio = dance_value / 100.0
    
    # 比率に基づいてカラーを計算
    color = (
        int(blue[0] + (red[0] - blue[0]) * ratio),
        int(blue[1] + (red[1] - blue[1]) * ratio),
        int(blue[2] + (red[2] - blue[2]) * ratio)
    )
    
    return color

def update(time_index):
    plot1.clear()
    scatter.clear()
    
    positions = []
    colors = []
    
    for df, checkbox in zip(dataframes, checkboxes):
        if checkbox.isChecked() and time_index < len(df):
            distances = df.iloc[time_index, 1:5].values
            dance_value = df.iloc[time_index, 5]
            x, y = trilaterate(distances)
            if not np.isnan(x) and not np.isnan(y):
                color = get_color(dance_value)  # グラデーションカラーを使用
                positions.append({'pos': (x, y), 'brush': pg.mkBrush(color=color)})
    
    scatter.addPoints(positions)
    plot1.addItem(scatter)
    
    if len(mean_dance) > time_index:
        mean_dance_value = mean_dance[time_index]
        marker.setData([time_index], [mean_dance_value])

    if user_interaction and playing:
        start_time_ms = time_index * 100
        pygame.mixer.music.play(start=start_time_ms / 1000)

def on_slider_change():
    global user_interaction
    if not playing:
        user_interaction = False
    update(slider.value())

def play():
    global playing, user_interaction
    playing = not playing
    user_interaction = True
    if playing:
        play_button.setText("Pause")
        pygame.mixer.music.play()
    else:
        play_button.setText("Play")
        pygame.mixer.music.stop()

def animate():
    if playing:
        current_value = slider.value()
        if current_value < slider.maximum():
            slider.setValue(current_value + 1)
        else:
            play()
    QtCore.QTimer.singleShot(100, animate)

def select_folder():
    folder_path = str(QFileDialog.getExistingDirectory(None, "Select Directory"))
    if folder_path:
        load_csv_files(folder_path)

def select_sound():
    global audio_file_path
    audio_file_path, _ = QFileDialog.getOpenFileName(None, "Select Sound File", "", "Audio Files (*.mp3 *.wav *.ogg)")
    if audio_file_path:
        load_sound_file(audio_file_path)

def load_sound_file(audio_file_path):
    audio_info = mediainfo(audio_file_path)
    sample_rate = int(audio_info['sample_rate'])
    channels = int(audio_info['channels'])
    pygame.mixer.init(frequency=sample_rate, channels=channels)
    pygame.mixer.music.load(audio_file_path)

def load_csv_files(folder_path):
    global dataframes, file_names, checkboxes, mean_dance
    dataframes = []
    file_names = []
    checkboxes = []
    checkbox_layout = checkbox_widget.layout()
    
    # 現在のチェックボックスを削除
    for i in reversed(range(checkbox_layout.count())):
        checkbox_layout.itemAt(i).widget().deleteLater()

    # 新しいファイルを読み込み
    csv_files = glob.glob(folder_path + '/*.csv')
    dataframes = [pd.read_csv(file) for file in csv_files]
    file_names = [file.split('/')[-1] for file in csv_files]

    # チェックボックスを作成
    for file_name in file_names:
        checkbox = QtWidgets.QCheckBox(file_name)
        checkbox.setChecked(True)
        checkboxes.append(checkbox)
        checkbox_layout.addWidget(checkbox)

    for checkbox in checkboxes:
        checkbox.stateChanged.connect(on_checkbox_change)

    if len(dataframes) > 0:
        slider.setRange(0, len(dataframes[0]) - 1)
        mean_dance = np.zeros(len(dataframes[0]))
        for i in range(len(mean_dance)):
            mean_dance[i] = np.mean([df.iloc[i, 5] for df in dataframes])
        curve.setData(mean_dance)  # 平均ダンス値をプロット
        update(0)

def on_checkbox_change(state):
    update(slider.value())

slider.valueChanged.connect(on_slider_change)
play_button.clicked.connect(play)
folder_button.clicked.connect(select_folder)
sound_button.clicked.connect(select_sound)

animate()
update(0)

if __name__ == '__main__':
    QtWidgets.QApplication.instance().exec_()
