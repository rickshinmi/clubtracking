import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
import glob
from matplotlib.animation import FuncAnimation
import pygame

# Trilateration function
def trilaterate(distances):
    beacon_positions = np.array([
        [0, 0],      # Beacon A
        [0, 10],     # Beacon B
        [10, 10],    # Beacon C
        [10, 0]      # Beacon D
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

# CSVファイルの読み込み
folder_path = '/Users/rickshinmi/Desktop/wearable_club/dummydata/'
csv_files = glob.glob(folder_path + '*.csv')
dataframes = [pd.read_csv(file) for file in csv_files]

# タイムスタンプのリストを作成
timestamps = pd.to_datetime(dataframes[0].iloc[:, 0])

# pygameの初期化
pygame.mixer.init()
audio_file_path = '/Users/rickshinmi/Desktop/wearable_club/soundfile/Mad Tribe - Fake Guru - 01 Fake Guru.mp3'
pygame.mixer.music.load(audio_file_path)

# プロットの設定
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.35)
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)
ax.plot([0, 10], [0, 0], 'k-')  # X軸
ax.plot([10, 10], [0, 10], 'k-')  # Y軸
ax.plot([10, 0], [10, 10], 'k-')  # X軸
ax.plot([0, 0], [10, 0], 'k-')  # Y軸

# カラーマップの設定
cmap = get_cmap('viridis')
norm = Normalize(vmin=0, vmax=100)
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

# スライダーの設定
ax_slider = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Time', 0, len(dataframes[0]) - 1, valinit=0, valstep=1, valfmt='%0.0f')

# 再生ボタンの設定
ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(ax_button, 'Play', color='lightgoldenrodyellow', hovercolor='0.975')

# フラグを管理するための変数
playing = False
user_interaction = False

def update(val):
    global user_interaction
    time_index = int(slider.val)
    ax.clear()  # グラフのクリア
    ax.plot([0, 10], [0, 0], 'k-')  # X軸
    ax.plot([10, 10], [0, 10], 'k-')  # Y軸
    ax.plot([10, 0], [10, 10], 'k-')  # X軸
    ax.plot([0, 0], [10, 0], 'k-')  # Y軸
    
    # 各データセットをプロット
    for df in dataframes:
        if time_index < len(df):
            distances = df.iloc[time_index, 1:5].values
            dance_value = df.iloc[time_index, 5]
            x, y = trilaterate(distances)
            
            if not np.isnan(x) and not np.isnan(y):  # NaNチェック
                color = sm.to_rgba(dance_value)
                ax.plot(x, y, 'o', color=color)  # 現在の位置をプロット

    ax.set_xlim([-1, 11])
    ax.set_ylim([-1, 11])
    plt.draw()
    # スライダーラベルをタイムスタンプに更新
    slider.valtext.set_text(timestamps[time_index].strftime('%Y-%m-%d %H:%M:%S'))

    # ユーザーがスライダーを動かした場合のみ音源再生の位置を変更
    if user_interaction:
        start_time_ms = time_index * 100  # 0.1秒ごとにサンプリングしているため
        pygame.mixer.music.play(start=start_time_ms / 1000)
        user_interaction = False

def on_slider_change(val):
    global user_interaction
    user_interaction = True
    update(val)

def play(event):
    global playing
    playing = not playing
    if playing:
        button.label.set_text('Pause')
        time_index = int(slider.val)
        start_time_ms = time_index * 100  # 0.1秒ごとにサンプリングしているため
        pygame.mixer.music.play(start=start_time_ms / 1000)
    else:
        button.label.set_text('Play')
        pygame.mixer.music.stop()

def animate(frame):
    global playing
    if playing:
        slider.set_val(slider.val + 1)
        if slider.val >= len(dataframes[0]) - 1:
            button.label.set_text('Play')
            playing = False

button.on_clicked(play)
slider.on_changed(on_slider_change)

# カラーバーを追加
cbar_ax = plt.axes([0.1, 0.02, 0.65, 0.03])
cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Dance Value')

# 初期ラベルをタイムスタンプに設定
slider.valtext.set_text(timestamps[0].strftime('%Y-%m-%d %H:%M:%S'))

ani = FuncAnimation(fig, animate, frames=np.arange(0, len(dataframes[0])), interval=100)

plt.show()
