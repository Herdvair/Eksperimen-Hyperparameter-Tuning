# Eksperimen Hyperparameter Tuning : Proyek Analisis & Optimasi Model

# 💡 Latar Belakang Proyek
Proyek ini bertujuan untuk membangun model Machine Learning yang optimal untuk memprediksi Customer Churn (pelanggan yang berhenti berlangganan) pada layanan telekomunikasi. Saya melakukan eksplorasi data, pembersihan data, dan pengujian model klasifikasi (Decision Tree, Random Forest, XG-Boost, dan LightGBM) dengan teknik Hyperparameter Tuning untuk mencapai performa prediksi terbaik. Selain itu, saya bereksperimen dengan membangun model regresi dengan target yang berbeda yaitu *Monthly Charges* atau *Biaya Bulanan* untuk menguji performa model regresi dengan hyperparameter tuning terhadap dataset ini. 

# 🛠️ Tahapan Proyek
1. Data Cleaning & Preprocessing
   - Pengecekan Missing Value: Sebagian besar kolom pada dataset ini, memiliki missing value yang beragam. Dapat dilihat melalui tabel dibawah ini:

| Kolom | Total Missing Values | 
| :--- | :---: | 
| customerID | 9 |
| Gender | 16 | 
| SeniorCitizen | 12 | 
| Partner | 16 |
| Dependents | 19 | 
| Tenure | 17 |
| Contract | 13 |
| PaperlessBilling | 13 |
| PaymentMethod | 11 |
| MonthlyCharges | 10 |

  - Penanganan Missing Value:
      - Dilakukan penghapusan baris yang terindikasi nilai NaN pada kolom customerID karena tidak dilakukan metode imputate karena ditakutkan adanya duplikasi data.
      - Dilakukan handling menggunakan median karena distribusi datanya terlihat asimetris pada kolom SeniorCitizen, Tenure, dan MonthlyCharges.
      - Dilakukan handling menggunakan mean karena distribusi datanya terlihat simetris/normal pada kolom Gender, Partner, Dependents, PaperlessBilling, PaymentMethod.

  - Encoding Data Kategori: Melakukan encoding data kategori menggunakan One Hot Encoding pada kolom Gender, Partner, Dependents, PaperlessBilling, PaymentMethod, Contract sebelum melakukan pemodelan.  

2. Exploratory Data Analysis (EDA)

  - **Distribusi Kolom Kategori Berdasarkan Churn** : Secara keseluruhan, proporsi pelanggan yang tidak churn lebih mendominasi di semua kategori. Namun, terdapat pola pada beberapa variabel yang menunjukkan bahwa pelanggan dengan kontrak month to month dan pelanggan dengan metode pembayaran electronic check lebih rentan terhadap churn (meskipun lebih tinggi tidak churn namun harus hati-hati).
  - **Distribusi Kolom Numerik Berdasarkan Churn** : Berdasarkan visualiasi diatas, pelanggan churn umumnya memiliki tenure yang singkat, biaya bulanan yang lebih tinggi dan total charges yang rendah (bisa jadi karena pelanggan tersebut hanya singkat menggunakan layanan). Hal ini mengindikasikan bahwa resiko churn lebih besar pada pelanggan baru dengan tagihan bulanan yang lebih tinggi, selain itu lama berlangganan menjadi salah satu faktor penting dalam meningkatkan loyalitas.
  - **Analisis Korelasi Numerik (Heatmap)**: Berdasarkan heatmap korelasi, terlihat bahwa tenure memiliki korelasi negatif dengan churn (-0.35), artinya semakin lama pelanggan berlangganan, semakin kecil kemungkinan mereka untuk churn. Sementara itu, total charges berkorelasi kuat dengan tenure (0.83) dan juga cukup tinggi dengan monthly charges (0.65), hal ini karena total charges merupakan akumulasi dari lama berlangganan dan biaya bulanan. Variabel senior citizen memiliki korelasi rendah terhadap churn (0.15), sehingga faktor usia lanjut tidak terlalu berpengaruh signifikan.
  - **Analisis Korelasi Kategorikal (Chi-Square & Cramer's V)**: Berdasarkan hasil uji Chi-Square, variabel Gender tidak memiliki hubungan signifikan dengan churn (p-value 0.5147, effect sangat lemah). Sementara itu, variabel Partner, Dependents, dan PaperlessBilling menunjukkan hubungan signifikan, namun dengan effect yang lemah terhadap churn. Variabel Contract dan PaymentMethod memiliki hubungan signifikan dengan effect yang lebih kuat, di mana Contract menunjukkan effect sedang (0.4092) dan PaymentMethod juga effect sedang (0.3024). Dengan demikian, faktor jenis kontrak dan metode pembayaran merupakan variabel kategori yang paling berpengaruh terhadap churn, sedangkan gender tidak berpengaruh.

3. Feature Engineering & Handling Imbalance
  - Encoding: Fitur-fitur kategorikal dikonversi menjadi format numerik menggunakan One Hot Encoding (OHE) dan Label Encoder.
  - Resampling: Untuk mengatasi imbalance data, digunakan teknik SMOTEENN (Synthetic Minority Oversampling Technique Edited Nearest Neighbor.

4. Multicolleniarity Check (Only Regresi Linear and Logistic Regression)
   - **Regresi Linear** : Menghapus kolom total charges (skor vif : 3.760976) dan tenure (skor vif : 5.489486), karena memiliki vif score yang tinggi sehingga diindikasikan bahwa kedua variabel tersebut memiliki keterkaitan satu sama lain.
   - **Logistic Regression** : Menghapus kolom total charges (skor vif : 9.828665) dan tenure (skor vif : 7.468032), karena memiliki vif score yang tinggi sehingga diindikasikan bahwa kedua variabel tersebut memiliki keterkaitan satu sama lain.

# 🤖 Pemodelan & Hasil

1. Pada eksperimen ini, dilakukan **uji performa model regresi linear** dalam prediksi MonthlyCharges (Biaya Bulanan). Dilakukan percobaan menggunakan based model, ridge, dan lasso untuk melihat hasil pemodelan regresi ini, berikut hasilnya:
   
| Metrik Evaluasi | Base Model (Train) | Ridge (Train) | Lasso (Train) | Base Model (Test) | Ridge (Test) | Lasso (Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **R²** | 0.27053129951604094 | 0.2704773247654042 | 0.27052313438345177 | 0.24407825420749474 | 0.24439249997095924 | 0.24419914192242553 |
| **RMSE** | 25.557194260554116 | 25.558139755316482 | 25.55733729430982 | 26.56407538210124 | 26.55855330529206 | 26.561951221242214 |
| **MAE** | 21.129077869469917 | 21.15388722779473 | 21.134724992201612 | 22.144893962256386 | 22.167627517731685 | 22.14880327329652 |
| **MAPE** | 0.521287973279167 | 0.5225157879757178 | 0.5215399891983812 | 0.5731654844239584 | 0.5742438572564526 | 0.57336602737061 |

Kesimpulan : 

Saya melakukan eksperimen untuk memprediksi Monthly Charges karena target Churn bersifat kategorikal, bukan kontinu. Namun, hasil dari base model, Ridge, maupun Lasso menunjukkan bahwa performa regresi terhadap data tersebut masih rendah (R² hanya sekitar 24–27%), jauh dari harapan. Hal ini mengindikasikan bahwa dataset kurang cocok digunakan untuk pemodelan regresi linear sederhana, baik dengan regularisasi maupun tanpa regularisasi. Kemungkinan penyebabnya adalah keterbatasan fitur dalam menjelaskan variasi Monthly Charges, sehingga model tidak mampu mencapai performa yang optimal meskipun dilakukan hyperparameter tuning.

2. Pada eksperimen ini, dilakukan **uji performa based model dari Logistic Regression, Decision Tree, Random Forest, XG-Boost, LightGBM, KNN** dalam prediksi Churn. Hal ini untuk identifikasi performa based model yang terbaik untuk dilakukan analisis lebih lanjut, berikut hasilnya:
   
| Metrik Evaluasi | Logistic Regression (Train) | Decision Tree (Train) | Random Forest (Train) | XG-Boost (Train) | LightGBM (Train) | KNN (Train) | Logistic Regression (Test) | Decision Tree (Test) | Random Forest (Test) | XG-Boost (Test) | LightGBM (Test) | KNN (Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Precision | 0.648760 | 1.000000 | 0.997330 | 0.887377 | 0.807630 | 0.723590 | 0.583916 | 0.460759 | 0.560976 | 0.586538 | 0.598684 | 0.526144 |
| F1-Score | 0.578909 | 0.995989 | 0.996000 | 0.831150 | 0.727871 | 0.659906 | 0.513846 | 0.479578 | 0.494624 | 0.541420 | 0.544910 | 0.480597 |
| AUC-ROC | 0.839759 | 0.999988 | 0.999973 | 0.974628 | 0.942349 | 0.900474 | 0.823635 | 0.649405 | 0.803801 | 0.821605 | 0.826449 | 0.744090 |
| Recall | 0.522636 | 0.992011 | 0.994674 | 0.781625 | 0.662450 | 0.606525 | 0.458791 | 0.500000 | 0.442308 | 0.502747 | 0.500000 | 0.442308 |

Kesimpulan : 

Model XG-Boost dan LightGBM menunjukkan performa cukup stabil, dapat dilakukan analisis lebih lanjut

3. Pada eksperimen ini, dilakukan analisis lanjutan pada model XG-Boost dan LightGBM dalam prediksi churn. Hal ini untuk perbandingan model yang terbaik dalam memprediksi churn berdasarkan strategi bisnis yang dituju. Berikut hasilnya:
   
| Model | Tahap | Dataset | Precision | Recall | F1-Score | ROC AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **XG-Boost** | Based Model | Train | 0.887377 | 0.781625 | 0.831150 | 0.974628 |
|  |  | Test | 0.586538 | 0.502747 | 0.541420 | 0.821605 |
|  | Hyperparameter Tuning | Train | 0.718724 | 0.569907 | 0.635722 | 0.883236 |
|  |  | Test | 0.601375 | 0.480769 | 0.534351 | 0.831445 |
|  | Hyperparameter Tuning + SMOTEENN | Train | 0.491282 | 0.844208 | 0.621112 | 0.836510 |
|  |  | Test | 0.470126 | 0.821429 | 0.598000 | 0.823760 |
| **LightGBM** | Based Model | Train | 0.807630 | 0.662450 | 0.727871 | 0.942349 |
|  |  | Test | 0.598684 | 0.500000 | 0.544910 | 0.826449 |
|  | Hyperparameter Tuning | Train | 0.710833 | 0.567909 | 0.631384 | 0.874526 |
|  |  | Test | 0.611296 | 0.505495 | 0.553383 | 0.835289 |
|  | Hyperparameter Tuning + SMOTEENN | Train | 0.710833 | 0.567909 | 0.631384 | 0.874526 |
|  |  | Test | 0.611296 | 0.505495 | 0.553383 | 0.835289 |

Kesimpulan: 
   - Diantara kedua model ini yaitu XG-Boost dan LightGBM. Semua bagus untuk pemodelan churn, namun dengan strategi bisnis berbeda. Jika strateginya untuk deteksi dini agar pelanggan churn tidak terlewat, maka model XG-Boost cocok karena recall nya lebih tinggi yaitu (0.82) dibandingkan Light-GBM (0.79). Sebaliknya, jika strateginya mengarah keseimbangan antara precision dan recall untuk mengurangi prediksi churn yang berlebihan maka model LightGBM cocok karena precision nya (0.53) sedangkan XG-Boost (0.47).
   - Berdasarkan hasil eksperimen dengan XGBoost:
        - Model based awal menunjukkan performa cukup baik dengan F1-score test sebesar (0.54), precision (0.58) dan AUC (0.82), namun recall masih rendah (0.50) sehingga model kurang optimal dalam menangkap pelanggan yang benar-benar churn.
        - Setelah dilakukan hyperparameter tuning tanpa SMOTEENN, performa test sedikit membaik pada F1-score (0.53), dan AUC (0.83), meskipun recall masih rendah (0.48) dan precision (0.49), menandakan bahwa model ini harus dioptimalisasi.
        - Penggunaan SMOTEENN dengan hyperparameter tuning memberikan performa lebih seimbang, dengan peningkatan signifikan recall (0.82) dan F1-score (0.59), meskipun precision sedikit menurun (0.47).
        - Artinya, model lebih sensitif dalam mendeteksi churn sehingga sebagian pelanggan loyal mungkin salah teridentifikasi sebagai berisiko churn. Namun, dalam konteks strategi bisnis, pendekatan ini lebih menguntungkan karena memungkinkan perusahaan melakukan deteksi dini agar tidak kehilangan pelanggan.
        
 - Berdasarkan hasil eksperimen dengan LightGBM:
         - Model based awal menunjukkan performa cukup baik dengan F1-score test sebesar (0.54), precision (0.59) dan (AUC 0.82), namun recall masih rendah (0.50) sehingga model kurang optimal dalam menangkap pelanggan yang benar-benar churn.
         - Setelah dilakukan hyperparameter tuning tanpa SMOTEENN, performa test sedikit membaik pada F1-score (0.55), precision (0.61) dan AUC (0.83), meskipun recall masih rendah (0.50), menandakan bahwa diperlukan optimalisasi.
         - Ketika SMOTEENN ditambahkan, recall meningkat signifikan menjadi (0.79) dengan F1-score (0.63), meskipun precision sedikit menurun (0.53), sehingga model menjadi cukup sensitif dalam mendeteksi churn. Secara keseluruhan, LightGBM dengan SMOTEENN memberikan hasil yang seimbang untuk kasus churn tergantung pada tujuan bisnis.

# 🎯 Potensi Dampak Bisnis
1. Model XG-Boost
   - **Berdasarkan perspektif bisnis** : Model ini cukup baik dalam mendeteksi pelanggan yang tidak churn yaitu (704). Sedangkan deteksi pelanggan yang churn yang berhasil diprediksi dengan benar yaitu sebanyak (299). Namun, model masih menghasilkan cukup banyak prediksi pelanggan yang churn namun aslinya tidak churn yaitu (337), artinya model ini lebih sensitif terhadap churn, sehingga cocok untuk strategi bisnis yang memprioritaskan deteksi dini pelanggan yang beresiko churn meskipun ada konsekuensi salah sasaran. Disisi lain, sebanyak 65 pelanggan churn yang tidak dideteksi oleh model sehingga model melabelinya sebagai tidak churn
   -**Berdasarkan kurva ROC-AUC** : Berdasarkan grafik ROC Curve hasil tuned model XGBoost, terlihat bahwa kurva hijau berada jauh di atas garis merah diagonal (baseline random guess), menandakan bahwa model memiliki kemampuan klasifikasi yang jauh lebih baik dibanding tebakan acak. Nilai AUC sebesar 0.8237 menunjukkan performa model yang cukup baik dalam membedakan antara pelanggan churn dan tidak churn. Semakin mendekati nilai 1, semakin baik kemampuan model. Dengan AUC di atas 0.8, model ini dapat dikatakan cukup andal untuk digunakan dalam deteksi churn, meskipun harus dilakukan optimaliasi kembali agar performa model menjadi lebih baik lagi.

2. Model LightGBM
   - **Berdasarkan perspektif bisnis** : Model ini cukup baik dalam mendeteksi pelanggan yang tidak churn yaitu (768). Sedangkan deteksi pelanggan yang churn yang berhasil diprediksi dengan benar yaitu sebanyak (271). Namun, model masih cukup dalam prediksi pelanggan yang churn namun aslinya tidak churn yaitu (273), artinya model ini tidak terlalu sensitif terhadap churn, tidak seperti model XG-Boost. Disisi lain, sebanyak (93) pelanggan churn yang tidak dideteksi oleh model sehingga model melabelinya sebagai tidak churn.
   - **Berdasarkan kurva ROC-AUC** : Berdasarkan grafik ROC Curve hasil tuned model LightGBM, kurva hijau berada cukup jauh di atas garis merah diagonal (baseline random guess), menandakan bahwa model memiliki kemampuan klasifikasi yang lebih baik dibanding tebakan acak. Nilai AUC sebesar 0.8210 menunjukkan performa model yang cukup baik dalam membedakan antara pelanggan churn dan tidak churn. Dengan nilai AUC di atas 0.8, LightGBM bisa dikategorikan sebagai model yang andal, meskipun performanya relatif mirip dengan XGBoost. Artinya, model ini cukup efektif digunakan dalam deteksi churn, meski masih harus dilakukan optimalisasi lagi agar mendapatkan performa model yang lebih baik.
  
**Dapat disimpulkan bahwa** : 

- XG-Boost lebih sensitif dalam mendeteksi pelanggan yang berisiko churn → cocok untuk strategi preventif (deteksi dini).
- LightGBM lebih konservatif, menghasilkan lebih sedikit false positive → cocok untuk strategi retensi yang lebih selektif.
