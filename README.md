# Laporan Proyek Machine Learning - Jauhar Mumtaz

## Domain Proyek

<p align="justify">
Nilai kelarutan dalam air molekul organik merupakan salah satu kunci sifat fisik dalam dunia medis. Karena berbanding lurus dengan absorpsi yang merupakan parameter utama distribusi senyawa aktif biologi dalam makhluk hidup dan lingkungan, sehingga berpengaruh pada potensi bioavailability, efektifitas dan daya jual senyawa aktif tersebut.
<p>

Pengukuran kelarutan dalam air dengan akurasi tinggi tentu membutuhkan *cost* yang tidak kecil, mulai dari waktu, instruments, kelihaian penguji, serta sampel fisik yang terbatas. Beberapa metode perhitungan kelarutan dalam air (S) telah dikembangkan seperti [*General Solubility Equation (GSE)*](https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.4c00685) dengan estimasi kelarutan dalam air (S) sebagai fungsi dari titik lebur (T) dan koefisien partisi oktanol-air (K):

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?log(S)=-0.01*(T-25%20C)-log(K)+0.50" alt="Equation" width="350">
</p>

<div style="background-color: #f0f0f0; padding: 10px; text-align: center; display: inline-block; margin: 0 auto;">
  <img src="https://latex.codecogs.com/svg.image?log(S)=-0.01*(T-25%20C)-log(K)+0.50" alt="Equation" width="350">
</div>

<p align="center">
  <div style="background-color: #f0f0f0; padding: 10px; text-align: center; display: inline-block;">
    <img src="https://latex.codecogs.com/svg.image?log(S)=-0.01*(T-25%20C)-log(K)+0.50" alt="Equation" width="350">
  </div>
</p>


Nilai partisi oktanol (K) dapat ditentukan berdasarkan struktur senyawa, namun penentuan titik lebur (T) masih memerlukan pengukuran lab. Sehingga metode GSE cocok untuk penentuan kelarutan dalam air suatu molekul jika tersedia data titik lebur-nya (T), sehingga metode yang dapat memanfaatkan struktur molekul untuk estimasi perlu dikembangkan.

Metode lain yang telah dikembangkan dengan menggunakan model machine learning yaitu [*Estimated Solubility (ESOL)*](https://pubs.acs.org/doi/abs/10.1021/ci034243x) yang memanfaatkan delapan parameter yang diekstrak menggunakan *molecular descriptor* seperti *clogP*, *molecular weight (molWT)*, *rotatable bond (rb)*, *aromatic proportion (ap)*, *non-carbon proportion*, *H-bond donor (hbd)*, *H-bond acceptor (hba)*, dan *polar surface area (psa)*. Berdasarkan 2874 data latih ESOL menghasilkan estimasi yang lebih *robust* dibandingkan GSE dengan nilai:

| | *R*² | SE | MAE |
| - | - | - | - |
| ESOL | 0.69 | 1.01 | 0.75 |
| GSE | 0.67 | 1.05 | 0.47 |

Metode ESOL juga menyimpulkan bahwa parameter paling signifikan yaitu *clogP* diikuti *molecular weight (molWT)*, *aromatic proportion (ap)*, dan *rotatable bond (rb)*. Dengan seiringnya perkembangan jaman, machine learning telah berkembang mulai dari beragamnya database, hyperparameter tuning, dan model, penulis ingin melanjutkan perkembangan estimasi nilai kelarutan molekul dalam air menggunakan dataset yang lebih besar seperti dataset [SMILES-enumeration-datasets](https://github.com/summer-cola/smiles-enumeration-datasets) dengan melakukan descriptor 0D, 1D, 2D, dam 3D dengan total 31 parameter yang digunakan sebagai input beberapa model regressor berbasis machine learning beserta deep learning seperti Neural Network (NN), K-Nearest Neighbors (KNN), Random Forest (RF), Support Vector Regressor (SVR), Elastic Net (EN), Decision Tree (DT), Extreme Gradient Boosting (XGBoost), Extra Trees (ET), dan Light Gradient-Boosting Machine (LightGBM). Tiga model terbaik berdasarkan nilai *Mean Absolute Error* (MAE) dan *Standard Error* (SE) rendah beserta Koefesien Determinasi (R²) akan dilakukan interpretasi model menggunakan SHapley Additive exPlanations (SHAP) sehingga didapatkan beberapa parameter paling signifikan.

## Business Understanding
### Problem Statements

*   Memprediksi kelarutan dalam air `LogS` suatu molekul `*drug-like*` merupakan tahap utama dalam dunia `*drug discovery*` yang mana dapat mempengaruhi efisiensi dan pengembangan obat. Apakah prediksi `LogS` dapat dilakukan menggunakan model *Machine Learning* yang tersedia ataupun *Deep Learning* sederhana dengan menggunkan fitur yang diekstrak hanya dari anotasi `SMILES` suatu molekul?
*   Di antara berbagai model *Machine Learning* yang tersedia ataupun *Deep Learning* sederhana, model manakah yang memiliki nilai *Mean Absolute Error* (MAE) dan *Standard Error* (SE) rendah beserta Koefesien Determinasi (R²) tinggi dalam memprediksi `LogS` berdasarkan fitur-fitur yang digunakan?
*   Dari delapan fitur yang digunakan dalam publikasi [ESOL](https://pubs.acs.org/doi/abs/10.1021/ci034243x#) yang dapat diekstraksi menggunakan deskriptor dari `SMILES`, fitur mana yang paling berpengaruh terhadap nilai `LogS`? Apakah terdapat fitur lain yang 

### Goals

*   Mengetahui prediksi LogS hanya dari ekstrakasi fitur dari `SMILES` dapat dilakukan menggunakan model *Machine Learning* yang tersedia ataupun *Deep Learning* sederhana.
*   Menentukan model *Machine Learning* yang tersedia ataupun Deep Learning sederhana dengan error terkecil untuk memprediksi nilai `LogS` berdasarkan fitur yang digunakan.
*   Mengidentifikasi fitur yang memiliki pengaruh terbesar terhadap nilai `LogS` (kelarutan molekul dalam air).

### Solution statements
*   Melakukan prediksi `LogS` dengan menggunkan fitur yang diekstrak hanya dari anotasi `SMILES` suatu molekul menggunakan model *Machine Learning* yang tersedia ataupun *Deep Learning* sederhana.
*   Menguji dan mengevaluasi beberapa model dengan *hyperparameter* yang telah ditetapkan sebelumnya, dan menetapkan model terbaik berdasarkan metrik *Mean Absolute Error* (MAE), *Standard Error* (SE), dan koefesien Determinasi (R²) tinggi .
*   Mengekstrak bobot fitur dari beberapa model.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

