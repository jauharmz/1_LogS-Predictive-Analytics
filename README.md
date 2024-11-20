# Laporan Proyek Machine Learning - Jauhar Mumtaz

## Domain Proyek

<p align="justify">
Nilai kelarutan dalam air molekul organik merupakan salah satu kunci sifat fisik dalam dunia medis. Karena berbanding lurus dengan absorpsi yang merupakan parameter utama distribusi senyawa aktif biologi dalam makhluk hidup dan lingkungan, sehingga berpengaruh pada potensi bioavailability, efektifitas dan daya jual senyawa aktif tersebut.
<p>

Pengukuran kelarutan dalam air dengan akurasi tinggi tentu membutuhkan *cost* yang tidak kecil, mulai dari waktu, instruments, kelihaian penguji, serta sampel fisik yang terbatas. Beberapa metode perhitungan kelarutan dalam air (S) telah dikembangkan seperti [*General Solubility Equation (GSE)*](https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.4c00685) dengan estimasi kelarutan dalam air (S) sebagai fungsi dari titik lebur (T) dan koefisien partisi oktanol-air (K):

<p align="center">
  <img src="./images/logS_GSE_dark.png">
</p>


Nilai partisi oktanol (K) dapat ditentukan berdasarkan struktur senyawa, namun penentuan titik lebur (T) masih memerlukan pengukuran lab. Sehingga metode GSE cocok untuk penentuan kelarutan dalam air suatu molekul jika tersedia data titik lebur-nya (T), sehingga metode yang dapat memanfaatkan struktur molekul untuk estimasi perlu dikembangkan.

Metode lain yang telah dikembangkan dengan menggunakan model machine learning yaitu [*Estimated Solubility (ESOL)*](https://pubs.acs.org/doi/abs/10.1021/ci034243x) yang memanfaatkan delapan parameter yang diekstrak menggunakan *molecular descriptor* seperti *clogP*, *molecular weight (molWT)*, *rotatable bond (rb)*, *aromatic proportion (ap)*, *non-carbon proportion*, *H-bond donor (hbd)*, *H-bond acceptor (hba)*, dan *polar surface area (psa)*. Berdasarkan 2874 data latih ESOL menghasilkan estimasi yang lebih *robust* dibandingkan GSE dengan nilai:

| | *R*² | SE | MAE |
| - | - | - | - |
| ESOL | 0.69 | 1.01 | 0.75 |
| GSE | 0.67 | 1.05 | 0.47 |

Metode ESOL juga menyimpulkan bahwa parameter paling signifikan yaitu *clogP* diikuti *molecular weight (molWT)*, *aromatic proportion (ap)*, dan *rotatable bond (rb)*. Dengan seiringnya perkembangan jaman, machine learning telah berkembang mulai dari beragamnya database, hyperparameter tuning, dan model, penulis ingin melanjutkan perkembangan estimasi nilai kelarutan molekul dalam air menggunakan dataset yang lebih besar dan hanya mengandalkan variabel `smiles` dan `log S` sebagai data `input` dan `label` utama degan menggunakan dataset [SMILES-enumeration-datasets](https://github.com/summer-cola/smiles-enumeration-datasets) dengan melakukan descriptor 0D, 1D, 2D, dam 3D pada variable `smiles` dan didapatkan total 31 parameter yang akan digunakan sebagai input beberapa model regressor berbasis machine learning beserta deep learning seperti Neural Network (NN), K-Nearest Neighbors (KNN), Random Forest (RF), Support Vector Regressor (SVR), Elastic Net (EN), Decision Tree (DT), Extreme Gradient Boosting (XGBoost), Extra Trees (ET), dan Light Gradient-Boosting Machine (LightGBM). Tiga model terbaik berdasarkan nilai *Mean Absolute Error* (MAE) dan *Standard Error* (SE) rendah beserta Koefesien Determinasi (R²) akan dilakukan interpretasi model menggunakan SHapley Additive exPlanations (SHAP) sehingga didapatkan beberapa parameter paling signifikan.

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
Dataset yang digunakan untuk memprediksi nilai `log S` suatu molekul diambil dari dataset [GitHub](https://github.com/summer-cola/smiles-enumeration-datasets) yang dipublish oleh `summer-cola` dengan nama repository [`SMILES-enumeration-datasets`](https://github.com/summer-cola/smiles-enumeration-datasets). Repository ini berisi beberapa dataset berisi sifat fisik suatu molekul seperti `log D`, `log P`, dan `log S`. Untuk dataset yang digunakan pada prediksi kali ini berada pada directory `logS` dengan nama file `traintest.csv` yang berisi 7954 baris data.

### Informasi Keterangan Variabel pada Dataset.

Dataset memiliki 8 variabel dengan keterangan sebagai berikut.

| Variabel | Deskripsi | Nilai |
| - | - | - |
| Unnamed: 0 | Indeks otomatis yang dihasilkan saat data diimpor. | 0 |
| Compound ID | ID unik untuk mengidentifikasi setiap senyawa dalam dataset. | C4659 |
| InChIKey | Kode alfanumerik yang merupakan versi singkat dari InChI (International Chemical Identifier) untuk identifikasi molekul unik secara global. | WIKXJKUZYYOTBP-UHFFFAOYSA-N |
| SMILES  | Simplified molecular input line entry system, bentuk notasi untuk deskripsi struktur molekul menggunakan *short ASCII strings*. | CCCCC(COC(=O)N)(COC(=O)NC(C)C)C |
| logS | Nilai logaritmik dari kelarutan dalam air (S), yang mengindikasikan seberapa larut suatu senyawa dalam air. | -3.633501683 |
| logP | Nilai logaritmik dari koefisien partisi oktanol-air (P), yang menunjukkan lipofilisitas atau kecenderungan senyawa untuk larut dalam lemak atau air. | 3.504 |
| MW  | Massa molekul (Molecular Weight), yaitu total massa atom dari molekul dalam satuan dalton (Da). | 274.357 |
| smi | Representasi alternative SMILES | C C C C C ( C O C ( = O ) N ) ( C O C ( = O ) N C ( C ) C ) C |


Dengan memanfaatkan `Descriptor` `0D`, `1D`, `2D`, dan `3D` variabel yang digunakan pada dari dataset yaitu `smiles` sebagai data `input mentah` dan `logS` sebagai `label`. Setelah dilakukan `Descriptor` pada varibel `smiles` didapatkan 31 variabel baru beserta ketaranganya sebagai berikut.

| Variabel | Deskripsi | Nilai |
| - | - | - |
| logS | LogS, nilai logaritmik kelarutan molekul (terutama obat) dalam air | -2.74 |
| molWt | Molecule weight, berat molekul | 170.92 |
| numAtoms | Jumlah atom berat (selain hidrogen dalam molekul). | 8 |
| molMR | Molecular refractivity, kemampuan molekul untuk membiaskan cahaya, terkait dengan polarizabilitas molekul. |  21.6 |
| rings | Jumlah cincin dalam struktur molekul | 0 |
| aromatic | Jumlah cincin dengan sifat aromatik dalam molekul. | 0 |
| ap | Aromatic proportion, rasio atom aromatik terhadap total atom. | 0.0 |
| chiralC | Jumlah pusat kiral (Karbon) dalam molekul. | 0 |
| logP | Koefisien partisi logaritmik mengukur kepolaran molekul. | 2.6496 |
| hbd | Jumlah donor ikatan hidrogen | 0 |
| hba | Jumlah akseptor ikatan hidrogen | 0 |
| rb | Rotatable bond, umlah ikatan rotasi| 1 |
| tpsa | Topological polar surface area, luas permukaan molekul yang bersifat polar | 0.0 |
| nh2 |  Jumlah gugus amina | 0 |
| oh | Jumlah gugus hidroksil | 0 |
| balabanJ | Indeks Balaban (Balaban J Index), ukuran kekompakan topologi molekul. | 4.020392 |
| bertzCT | Kompleksitas topologi Bertz (Bertz CT), ukuran kerumitan molekul berdasarkan struktur graf. | 67.01955 |
| hallKierAlpha |  Indeks Hall-Kier Alpha, terkait dengan bentuk molekul dan polarizabilitasnya. | 0.3 |
| ipc | Indeks polaritas informasi (Information Content Index), mengukur keragaman struktur molekul. |  21.306059 |
| chi1 | Chi Path Index 0, pengukuran topologi molekul berdasarkan jumlah dan jenis atom. | 7.0 |
| chi2 | Chi Path Index 1, pengukuran jalur molekul berdasarkan pola ikatan atom. |  3.25 |
| kappa1 | Indeks kappa molekuler 1, mengukur fleksibilitas molekul. | 8.3 |
| kappa2 | Indeks kappa molekuler 2, variasi lain untuk mengukur fleksibilitas molekul. |  1.91511 |
| kappa3 | Indeks kappa molekuler 3, variasi lebih lanjut dari pengukuran fleksibilitas.| 2.046098 |
|fractionCSP3 | Fraksi atom karbon dalam hibridisasi sp3 | 1.0|
| asphericity |  Asferisitas, pengukuran penyimpangan bentuk molekul dari bentuk bola sempurna. | 0.072556 |
| eccentricity | Eksentrisitas, pengukuran asimetri dalam distribusi atom molekul. | 0.785158 |
| inertialShapeFactor | Faktor bentuk inersia, yang menunjukkan bentuk molekul berdasarkan distribusi massa atom. | 0.003042 |
| radiusOfGyration | Jari-jari perputaran (Radius of Gyration), mengukur penyebaran atom dalam molekul relatif terhadap pusat massa. | 1.836359 |
| spherocityIndex | Indeks sferisitas, yang menunjukkan seberapa dekat bentuk molekul dengan bola. | 0.711911 |
| ncp | Proporsi non-karbon terhadap total atom dalam molekul. | 0.75 |
| ecfp | Extended Circular Fingerprints, representasi molekul berbasis bit. | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... ] |

Berikut merupakan keterangan tipe data pada dataset.

| Tipe Data | Variabel | Keterangan |
| - | - | - |
| Float | logS, molWt, molMR, ap, logP, tpsa, balabanJ, bertzCT, hallKierAlpha, ipc, chi0, chi1, kappa1, kappa2, kappa3, fractionCSP3, asphericity, eccentricity, inertialShapeFactor, radiusOfGyration, spherocityIndex, dan ncp | Data hasil kalkulasi matematis dan fraksi. |
| Integer | numAtoms, rings, aromatic, chiralC, hbd, hba, rb, nh2, dan oh | Data penjumlahan satuan |
| List | ecfp | Data berisi 2048 bit interpretasi molekul.  |


## Data Cleaning

Setelah dilakukan pengecekan data `Null`, `NaN` dan data duplikat hanya ditemukan `1` data duplikat sehingga dilakukan `drop` data dan didapatkan deskripsi statistik dataset sebagai berikut.

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logS</th>
      <th>molWt</th>
      <th>numAtoms</th>
      <th>molMR</th>
      <th>rings</th>
      <th>aromatic</th>
      <th>ap</th>
      <th>chiralC</th>
      <th>logP</th>
      <th>hbd</th>
      <th>...</th>
      <th>kappa1</th>
      <th>kappa2</th>
      <th>kappa3</th>
      <th>fractionCSP3</th>
      <th>asphericity</th>
      <th>eccentricity</th>
      <th>inertialShapeFactor</th>
      <th>radiusOfGyration</th>
      <th>spherocityIndex</th>
      <th>ncp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7954.000000</td>
      <td>7954.000000</td>
      <td>7954.000000</td>
      <td>7954.000000</td>
      <td>7954.000000</td>
      <td>7954.000000</td>
      <td>7954.000000</td>
      <td>7954.00000</td>
      <td>7954.000000</td>
      <td>7954.000000</td>
      <td>...</td>
      <td>7954.000000</td>
      <td>7954.000000</td>
      <td>7954.000000</td>
      <td>7954.000000</td>
      <td>7954.000000</td>
      <td>7954.000000</td>
      <td>7954.000000</td>
      <td>7.954000e+03</td>
      <td>7954.000000</td>
      <td>7954.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-2.981528</td>
      <td>292.151987</td>
      <td>19.181795</td>
      <td>75.840784</td>
      <td>1.975107</td>
      <td>1.195248</td>
      <td>0.352023</td>
      <td>0.97297</td>
      <td>1.912550</td>
      <td>1.239125</td>
      <td>...</td>
      <td>3.924647</td>
      <td>6.902837</td>
      <td>5.105968</td>
      <td>0.459514</td>
      <td>0.392271</td>
      <td>0.937606</td>
      <td>0.002492</td>
      <td>3.618187e+00</td>
      <td>0.250457</td>
      <td>0.270458</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.200720</td>
      <td>138.909559</td>
      <td>9.048712</td>
      <td>34.724211</td>
      <td>1.461655</td>
      <td>0.982673</td>
      <td>0.260995</td>
      <td>2.21534</td>
      <td>2.510816</td>
      <td>1.513059</td>
      <td>...</td>
      <td>2.528799</td>
      <td>4.207856</td>
      <td>22.793668</td>
      <td>0.301113</td>
      <td>0.193149</td>
      <td>0.065069</td>
      <td>0.010385</td>
      <td>1.462360e+00</td>
      <td>0.161920</td>
      <td>0.133211</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-16.259392</td>
      <td>16.043000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>-46.668600</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-27.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.469447e-18</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-4.259388</td>
      <td>197.190000</td>
      <td>13.000000</td>
      <td>51.231650</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.779440</td>
      <td>0.000000</td>
      <td>...</td>
      <td>2.475306</td>
      <td>4.088493</td>
      <td>2.171001</td>
      <td>0.235294</td>
      <td>0.239931</td>
      <td>0.913155</td>
      <td>0.000463</td>
      <td>2.823452e+00</td>
      <td>0.131913</td>
      <td>0.181818</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-2.824600</td>
      <td>273.798500</td>
      <td>18.000000</td>
      <td>72.044850</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.375000</td>
      <td>0.00000</td>
      <td>2.045950</td>
      <td>1.000000</td>
      <td>...</td>
      <td>3.534830</td>
      <td>6.038287</td>
      <td>3.400000</td>
      <td>0.428571</td>
      <td>0.374412</td>
      <td>0.957519</td>
      <td>0.001016</td>
      <td>3.473871e+00</td>
      <td>0.230156</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-1.489481</td>
      <td>359.713500</td>
      <td>23.000000</td>
      <td>93.985125</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.545455</td>
      <td>1.00000</td>
      <td>3.401700</td>
      <td>2.000000</td>
      <td>...</td>
      <td>4.818714</td>
      <td>8.679717</td>
      <td>5.344128</td>
      <td>0.666667</td>
      <td>0.532319</td>
      <td>0.981270</td>
      <td>0.002296</td>
      <td>4.223524e+00</td>
      <td>0.347257</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.580000</td>
      <td>1583.582000</td>
      <td>109.000000</td>
      <td>370.217200</td>
      <td>16.000000</td>
      <td>12.000000</td>
      <td>1.000000</td>
      <td>27.00000</td>
      <td>20.854600</td>
      <td>19.000000</td>
      <td>...</td>
      <td>72.265273</td>
      <td>62.805231</td>
      <td>1128.960000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.339204</td>
      <td>5.985842e+01</td>
      <td>0.999963</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>

Dari deskripsi data statistik di atas, dapat disimpulkan bahwa sebaran data `mean`, `Q2`, dan `std` yang bervariatif. Outlier diperiksa menggunakan 8 variabel yang digunakan pada penelitian `ESOL` sebagai berikut.

<p align="center">
  <img src="./images/outlier.png">
</p>

Interpretasi Outlier :
1. `log P` : Sebaran data terpusat pada rentang `0.8 - 3.5` dari rentang `-46.6 - 20.8`.
2. `molWt` : Sebaran data terpusat pada rentang `197.1 - 359.7` dari rentang `16 - 1583.5`.
3. `rb` : Sebaran data terpusat pada rentang `3 - 9` dari rentang `0 - 59`.
4. `ap` : Sebaran data terpusat pada rentang `0 - 0.5` dari rentang `0 - 1`.
5. `ncp` : Sebaran data terpusat pada rentang `0.1 - 0.3` dari rentang `0 - 1`.
6. `hbd` : Sebaran data terpusat pada rentang `0 - 2` dari rentang `0 - 19`.
7. `hba` : Sebaran data terpusat pada rentang `2 - 4` dari rentang `0 - 35`.
8. `tpsa` : Sebaran data terpusat pada rentang `25.3 - 71.4` dari rentang `25.3 - 601.8`.

Yang dapat diartikan bahwa outlier merupakan interpretasi nilai fisika molekuler berdasarkan strukturnya, sehingga dimungkinkan terdapatnya outlier dan tidak dilakukan penghapusan outlier yang dapat berakibat hilangnya sebagian besar dataset.

### Univariate - Numerical Features

### Multivariate - Numerical Features


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

