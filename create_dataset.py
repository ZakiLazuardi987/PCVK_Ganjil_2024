import os  # Mengimpor modul os untuk operasi sistem file
import pickle  # Mengimpor modul pickle untuk serialisasi objek

import mediapipe as mp  # Mengimpor modul mediapipe untuk deteksi tangan
import cv2  # Mengimpor modul OpenCV untuk pemrosesan gambar dan video
import matplotlib.pyplot as plt  # Mengimpor matplotlib untuk visualisasi (tidak digunakan dalam kode ini)

# Inisialisasi modul Mediapipe untuk deteksi tangan
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Menginisialisasi objek Hands dari Mediapipe dengan mode gambar statis dan tingkat kepercayaan deteksi minimum 0.3
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = "./data"  # Mendefinisikan direktori penyimpanan data

data = []  # Inisialisasi list untuk menyimpan data fitur tangan
labels = []  # Inisialisasi list untuk menyimpan label kelas

# Melakukan iterasi melalui setiap direktori (kelas) dalam direktori DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(
        os.path.join(DATA_DIR, dir_)
    ):  # Iterasi melalui setiap gambar dalam direktori kelas
        data_aux = []  # List sementara untuk menyimpan fitur dari satu gambar

        x_ = []  # List untuk menyimpan koordinat x dari landmark tangan
        y_ = []  # List untuk menyimpan koordinat y dari landmark tangan

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))  # Membaca gambar
        img_rgb = cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB
        )  # Mengonversi gambar dari BGR ke RGB

        results = hands.process(
            img_rgb
        )  # Memproses gambar dengan Mediapipe untuk mendeteksi tangan
        if (
            results.multi_hand_landmarks
        ):  # Memeriksa apakah ada landmark tangan yang terdeteksi
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(
                    len(hand_landmarks.landmark)
                ):  # Iterasi melalui setiap landmark tangan
                    x = hand_landmarks.landmark[
                        i
                    ].x  # Mendapatkan koordinat x dari landmark
                    y = hand_landmarks.landmark[
                        i
                    ].y  # Mendapatkan koordinat y dari landmark

                    x_.append(x)  # Menambahkan koordinat x ke list x_
                    y_.append(y)  # Menambahkan koordinat y ke list y_

                for i in range(
                    len(hand_landmarks.landmark)
                ):  # Normalisasi koordinat landmark tangan
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(
                        x - min(x_)
                    )  # Menyimpan koordinat x yang dinormalisasi ke data_aux
                    data_aux.append(
                        y - min(y_)
                    )  # Menyimpan koordinat y yang dinormalisasi ke data_aux

            data.append(data_aux)  # Menambahkan data fitur tangan ke list data
            labels.append(dir_)  # Menambahkan label kelas ke list labels

# Menyimpan data dan label ke file pickle
f = open("data.pickle", "wb")
pickle.dump({"data": data, "labels": labels}, f)
f.close()
