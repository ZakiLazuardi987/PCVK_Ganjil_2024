import pickle  # Mengimpor modul pickle untuk deserialisasi dan serialisasi objek

from sklearn.ensemble import (
    RandomForestClassifier,
)  # Mengimpor RandomForestClassifier dari scikit-learn
from sklearn.model_selection import (
    train_test_split,
)  # Mengimpor train_test_split untuk membagi dataset
from sklearn.metrics import (
    accuracy_score,
)  # Mengimpor accuracy_score untuk mengevaluasi akurasi model
import numpy as np  # Mengimpor numpy untuk operasi array

# Membuka file pickle yang berisi data dan label yang sudah diproses sebelumnya
data_dict = pickle.load(open("./data.pickle", "rb"))

# Mengonversi data dan label dari dictionary ke numpy array
data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

# Membagi dataset menjadi data latih (training) dan data uji (testing)
x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels,  # 20% dari data digunakan untuk pengujian, dengan stratifikasi berdasarkan label
)

# Menginisialisasi model RandomForestClassifier
model = RandomForestClassifier()

# Melatih model dengan data latih
model.fit(x_train, y_train)

# Memprediksi label untuk data uji
y_predict = model.predict(x_test)

# Menghitung akurasi prediksi
score = accuracy_score(y_predict, y_test)

# Menampilkan persentase sampel yang diklasifikasikan dengan benar
print("{}% of samples were classified correctly !".format(score * 100))

# Menyimpan model yang sudah dilatih ke file pickle
f = open("model.p", "wb")
pickle.dump({"model": model}, f)
f.close()
