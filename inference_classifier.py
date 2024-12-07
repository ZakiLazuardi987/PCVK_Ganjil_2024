import pickle  # Mengimpor modul pickle untuk deserialisasi objek
import cv2  # Mengimpor OpenCV untuk pemrosesan video dan gambar
import mediapipe as mp  # Mengimpor Mediapipe untuk deteksi tangan
import numpy as np  # Mengimpor numpy untuk operasi array
import pyttsx3  # Mengimpor pyttsx3 untuk mengeluarkan suara

# Membuka file pickle yang berisi model yang sudah dilatih
model_dict = pickle.load(open("./model.p", "rb"))
model = model_dict["model"]  # Mengambil model dari dictionary

cap = cv2.VideoCapture(0)  # Menginisialisasi video capture dari kamera (webcam)

# Inisialisasi modul Mediapipe untuk deteksi tangan
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Menginisialisasi objek Hands dari Mediapipe dengan mode gambar statis dan tingkat kepercayaan deteksi minimum 0.3
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary untuk label kelas
labels_dict = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z",
}

# Inisialisasi pyttsx3 untuk text-to-speech
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Mengatur kecepatan suara
engine.setProperty("volume", 1.0)  # Mengatur volume suara

# Variabel untuk menyimpan karakter yang telah diucapkan sebelumnya
last_predicted_character = None

# Loop utama untuk membaca frame dari kamera dan melakukan prediksi
while True:
    data_aux = []  # List sementara untuk menyimpan data fitur tangan
    x_ = []  # List untuk menyimpan koordinat x dari landmark tangan
    y_ = []  # List untuk menyimpan koordinat y dari landmark tangan

    ret, frame = cap.read()  # Membaca frame dari kamera

    H, W, _ = frame.shape  # Mendapatkan dimensi frame

    frame_rgb = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )  # Mengonversi frame dari BGR ke RGB

    results = hands.process(
        frame_rgb
    )  # Memproses frame dengan Mediapipe untuk mendeteksi tangan
    if (
        results.multi_hand_landmarks
    ):  # Memeriksa apakah ada landmark tangan yang terdeteksi
        for hand_landmarks in results.multi_hand_landmarks:
            # Menggambar landmark tangan dan koneksi pada frame
            mp_drawing.draw_landmarks(
                frame,  # gambar untuk menggambar
                hand_landmarks,  # output model
                mp_hands.HAND_CONNECTIONS,  # koneksi tangan
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Reset data_aux untuk setiap tangan
            data_aux = []
            x_ = []
            y_ = []

            for i in range(21):  # Hanya memproses 21 landmark tangan
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(21):  # Hanya memproses 21 landmark tangan
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # Normalisasi koordinat x
                data_aux.append(y - min(y_))  # Normalisasi koordinat y

            x1 = (
                int(min(x_) * W) - 10
            )  # Menghitung koordinat kiri atas dari bounding box
            y1 = (
                int(min(y_) * H) - 10
            )  # Menghitung koordinat kiri atas dari bounding box

            x2 = (
                int(max(x_) * W) - 10
            )  # Menghitung koordinat kanan bawah dari bounding box
            y2 = (
                int(max(y_) * H) - 10
            )  # Menghitung koordinat kanan bawah dari bounding box

            prediction = model.predict(
                [np.asarray(data_aux)]
            )  # Memprediksi kelas dari data fitur tangan

            predicted_character = labels_dict[
                int(prediction[0])
            ]  # Mendapatkan karakter prediksi dari label dictionary

            # Menggambar bounding box dan karakter prediksi pada frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(
                frame,
                predicted_character,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )

            # Mengeluarkan suara jika karakter prediksi berbeda dari karakter sebelumnya
            if predicted_character != last_predicted_character:
                engine.say(predicted_character)
                engine.runAndWait()
                last_predicted_character = predicted_character

    cv2.imshow(
        "frame", frame
    )  # Menampilkan frame dengan bounding box dan karakter prediksi
    cv2.waitKey(1)  # Menunggu selama 1 milidetik sebelum membaca frame berikutnya

cap.release()  # Melepaskan objek video capture
cv2.destroyAllWindows()  # Menutup semua jendela OpenCV
