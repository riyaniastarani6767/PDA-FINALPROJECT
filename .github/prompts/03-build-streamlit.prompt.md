
---
name: Build Streamlit
description: Bangun antarmuka Streamlit berdasarkan backend yang telah diverifikasi.
---

Bangun aplikasi Streamlit menggunakan backend proyek yang sudah terintegrasi.

Halaman yang dibutuhkan:

1. Prediksi
   - Input manual 30 fitur
   - Pemilihan sampel dataset
   - Upload CSV
   - Pilihan model Random Forest atau SVM
   - Hasil prediksi benign atau malignant
   - Probabilitas prediksi jika didukung model
   - Interpretasi model melalui SHAP atau feature importance sesuai implementasi yang tersedia

2. Riwayat Prediksi
   - Menampilkan riwayat dari SQLite
   - Informasi waktu, input, model, dan hasil
   - Detail prediksi

3. Evaluasi Model
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix
   - Visualisasi hasil evaluasi yang benar-benar berasal dari eksperimen

4. Informasi Dataset
   - Deskripsi dataset
   - Jumlah sampel
   - Distribusi kelas
   - Penjelasan fitur

Ketentuan:
- Gunakan desain yang bersih dan profesional.
- Jangan menampilkan angka hasil eksperimen yang belum diverifikasi.
- Jangan mengubah metodologi penelitian.
- Jangan melakukan training pada setiap rerun Streamlit.
- Validasi semua input.
- Batasi jumlah baris CSV sesuai hasil audit.
- Tampilkan disclaimer bahwa aplikasi bukan alat diagnosis klinis.

Sebelum implementasi, laporkan rencana file yang akan dibuat atau diubah.

Implementasikan secara bertahap dan uji setiap halaman.
