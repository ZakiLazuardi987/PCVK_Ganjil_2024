import os  # Mengimpor modul os untuk operasi sistem file
import cv2  # Mengimpor modul OpenCV untuk pemrosesan gambar dan video

DATA_DIR = "./data"  # Mendefinisikan direktori penyimpanan data
if not os.path.exists(DATA_DIR):  # Memeriksa apakah direktori DATA_DIR sudah ada
    os.makedirs(DATA_DIR)  # Jika tidak, membuat direktori DATA_DIR

number_of_classes = 26  # Mengatur jumlah kelas data yang akan dikumpulkan
dataset_size = 100  # Mengatur ukuran dataset untuk setiap kelas

cap = cv2.VideoCapture(0)  # Mengaktifkan webcam untuk menangkap video
for j in range(number_of_classes):  # Melakukan iterasi untuk setiap kelas
    if not os.path.exists(
        os.path.join(DATA_DIR, str(j))
    ):  # Memeriksa apakah direktori untuk kelas j sudah ada
        os.makedirs(
            os.path.join(DATA_DIR, str(j))
        )  # Jika tidak, membuat direktori untuk kelas j

    print(
        "Collecting data for class {}".format(j)
    )  # Menampilkan pesan pengumpulan data untuk kelas j

    done = False  # Inisialisasi variabel done sebagai False
    while True:  # Looping untuk menunggu input pengguna
        ret, frame = cap.read()  # Membaca frame dari webcam
        cv2.putText(  # Menambahkan teks pada frame
            frame,
            'Ready? Press "Q" ! :)',  # Teks yang ditampilkan
            (100, 50),  # Posisi teks pada frame
            cv2.FONT_HERSHEY_SIMPLEX,  # Font teks
            1.3,  # Ukuran font
            (0, 255, 0),  # Warna teks (hijau)
            3,  # Ketebalan teks
            cv2.LINE_AA,  # Tipe garis teks
        )
        cv2.imshow("frame", frame)  # Menampilkan frame dengan teks
        if cv2.waitKey(25) == ord("q"):  # Menunggu pengguna menekan tombol 'Q'
            break  # Jika 'Q' ditekan, keluar dari loop

    counter = 0  # Inisialisasi counter sebagai 0
    while (
        counter < dataset_size
    ):  # Looping untuk mengumpulkan gambar hingga mencapai dataset_size
        ret, frame = cap.read()  # Membaca frame dari webcam
        cv2.imshow("frame", frame)  # Menampilkan frame
        cv2.waitKey(25)  # Menunggu selama 25 milidetik
        cv2.imwrite(
            os.path.join(DATA_DIR, str(j), "{}.jpg".format(counter)), frame
        )  # Menyimpan frame sebagai file gambar
        counter += 1  # Menambah counter

cap.release()  # Melepaskan resource webcam
cv2.destroyAllWindows()  # Menutup semua jendela OpenCV yang terbuka
