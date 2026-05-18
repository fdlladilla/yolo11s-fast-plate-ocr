# =======================================================
# Sistem Gerbang Otomatis - Real-Time Webcam (Final OOP)
# Pipeline: YOLO11s + Kustom CCT OCR (Fast-Plate OCR)
# =======================================================

import os
import cv2
import re
import numpy as np
import yaml
import onnxruntime as ort
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

class SistemGerbangOtomatis:
    def __init__(self):
        # 1. Tentukan jalur file model secara absolut
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        yolo_path = os.path.join(self.base_dir, "best.pt")
        onnx_path = os.path.join(self.base_dir, "cct_xs_v1_global.onnx")
        config_path = os.path.join(self.base_dir, "config", "indonesian_plate_config.yaml")

        print("\n" + "="*50)
        print("🤖 Menginisialisasi Sistem Fast-Plate OCR Real-Time...")
        print("="*50)

        # 2. Validasi keberadaan file sebelum sistem dinyalakan
        if not os.path.exists(yolo_path) or not os.path.exists(onnx_path) or not os.path.exists(config_path):
            raise FileNotFoundError("❌ Error: Pastikan best.pt, cct_xs_v1_global.onnx, dan folder config sudah lengkap!")

        # 3. Muat Konfigurasi Karakter OCR
        print("-> Membaca konfigurasi alfabet Indonesia...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.alphabet = config['alphabet']

        # 4. Muat Model Deteksi & Pembaca
        print("-> Memuat Model YOLO11s...")
        self.yolo_model = YOLO(yolo_path)

        print("-> Memuat Sesi Kustom CCT OCR ONNX...")
        self.ocr_session = ort.InferenceSession(onnx_path)
        self.ocr_input_name = self.ocr_session.get_inputs()[0].name
        
        print("✅ Sistem Siap! Kamera akan segera menyala...")

    def bersihkan_teks(self, text):
        return re.sub(r'[^A-Z0-9]', '', text.upper()).strip()

    def baca_plat_nomor(self, potongan_plat):
        if potongan_plat.size == 0:
            return ""

        # Preprocessing khusus untuk menyamakan input model ONNX kustom (128x64)
        img_gray = cv2.cvtColor(potongan_plat, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (128, 64))
        img_rgb_ocr = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_input = np.expand_dims(img_rgb_ocr, axis=0)

        # Jalankan inferensi ONNX
        preds = self.ocr_session.run(None, {self.ocr_input_name: img_input})[0]
        best_path = np.argmax(preds, axis=-1)[0]
        
        # Susun text berdasarkan urutan alfabet dari kamus yaml
        hasil_mentah = "".join([self.alphabet[idx] for idx in best_path if idx < len(self.alphabet) and self.alphabet[idx] != '_'])
        return self.bersihkan_teks(hasil_mentah)

    def mulai_kamera(self, index_kamera=0):
        # Membuka webcam internal MacBook (0 biasanya adalah default FaceTime HD Camera)
        cap = cv2.VideoCapture(index_kamera)
        if not cap.isOpened():
            print("❌ Gagal membuka webcam Mac. Berikan izin akses kamera ke terminal/VS Code jika diminta.")
            return

        print("\n🎬 Webcam Menyala! Dekatkan plat nomor ke kamera.")
        print("⌨️ Tekan tombol 'q' pada jendela video untuk keluar.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Gagal menerima frame dari webcam.")
                break

            # Tahap 1: Deteksi lokasi plat menggunakan YOLO
            # Warna diubah ke RGB agar YOLO mendeteksi dengan akurasi optimal
            img_rgb_yolo = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.yolo_model(img_rgb_yolo, conf=0.4, verbose=False)
            boxes = results[0].boxes

            # Siapkan anotasi box di atas frame layar
            annotator = Annotator(frame, line_width=3)

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Potong bagian plat nomor saja
                img_cropped = frame[y1:y2, x1:x2]
                
                # Tahap 2: Oper hasil potong ke OCR kustom
                teks_plat = self.baca_plat_nomor(img_cropped)

                if teks_plat:
                    # Gambar kotak berwarna HIJAU jika berhasil mendeteksi dan membaca teks
                    annotator.box_label([x1, y1, x2, y2], label=teks_plat, color=(0, 255, 0))
                else:
                    # Gambar kotak berwarna MERAH jika kotak terdeteksi tapi teks gagal di-decode
                    annotator.box_label([x1, y1, x2, y2], label="Membaca...", color=(0, 0, 255))

            # Tampilkan frame video streaming ke layar pop-up Mac
            cv2.imshow("Sistem ANPR Fast-Plate OCR - Realtime Webcam", frame)

            # Batasan keluar loop: Tekan tombol huruf 'q' kecil pada keyboard
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Matikan kamera dan tutup semua pop-up jendela
        cap.release()
        cv2.destroyAllWindows()
        print("🏁 Sistem dimatikan dengan sukses.")

if __name__ == "__main__":
    try:
        sistem = SistemGerbangOtomatis()
        # Jika kamera bawaan Mac tidak menyala, ganti angka 0 di bawah menjadi 1 atau 2
        sistem.mulai_kamera(index_kamera=0)
    except Exception as e:
        print(e)