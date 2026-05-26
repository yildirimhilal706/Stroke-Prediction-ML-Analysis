# İnme (Stroke) Tahmin Modeli

Bu projede, kişilerin demografik ve sağlık verilerinden yola çıkarak inme (stroke) riski olup olmadığını tahmin eden bir makine öğrenmesi modeli geliştirildi. Veri seti **dengesiz** (yalnızca %4.87 pozitif vaka) olduğu için odak noktası yüksek doğruluk değil, **inme vakalarını kaçırmayan** bir model kurmaktı.

## Veri Seti

- **Kaynak:** Kaggle — Healthcare Stroke Dataset
- **Boyut:** 5110 satır, 12 kolon (temizleme sonrası 5109 × 11)
- **Hedef değişken:** `stroke` (0: yok, 1: var)
- **Sınıf dağılımı:** 4861 negatif, 249 pozitif → **%95.13 / %4.87**

Özellikler: `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`.

## İş Akışı

1. **Veri Temizleme**
   - `bmi` kolonundaki 201 eksik değer medyan (28.1) ile dolduruldu
   - `gender == 'Other'` olan tek satır çıkarıldı
   - `id` kolonu düşürüldü
2. **Keşifsel Veri Analizi (EDA)**
   - Sayısal değişkenler için histogram ve istatistiki özet
   - `age`, `avg_glucose_level`, `bmi` için stroke durumuna göre boxplot karşılaştırması
   - Kategorik değişkenler için stroke kırılımlı countplot'lar
   - Sayısal özellikler arası korelasyon ısı haritası
3. **Ön İşleme**
   - Kategorik değişkenler için One-Hot Encoding (`drop_first=True`)
   - `train_test_split` ile %80/%20 stratified ayrım
   - `StandardScaler` ile ölçeklendirme
4. **Modelleme** — üç farklı yaklaşım denendi ve karşılaştırıldı

## Denenen Modeller ve Sonuçlar

Tüm modeller `RandomForestClassifier` üzerine kuruldu. Pozitif sınıfa (stroke = 1) ait metrikler:

| Model | Precision (1) | Recall (1) | F1 (1) | Accuracy |
|---|---|---|---|---|
| Baseline (`class_weight='balanced_subsample'`) | 0.27 | 0.16 | 0.20 | 0.94 |
| **SMOTE + Random Forest** | **0.13** | **0.40** | **0.20** | **0.86** |
| SMOTE + GridSearchCV (recall odaklı) | 0.08 | 0.17 | 0.11 | 0.88 |

**Final model olarak SMOTE'lu Random Forest seçildi.** Sebebi: sağlık uygulamalarında pozitif vakayı kaçırmamak (yüksek recall) yanlış alarm vermekten daha kritik. Grid Search recall'ı düşürdüğü için baseline SMOTE modeli korundu.

### Final Model Hata Matrisi
Tahmin: Yok   Tahmin: Var
Gerçek: Yok        830           110
Gerçek: Var         25            17

Test setindeki 42 inme vakasının 17'si yakalandı, 25'i kaçırıldı.

## Öne Çıkan Özellikler

Modelin karar verirken en çok ağırlık verdiği değişkenler (feature importance):

1. `age`
2. `avg_glucose_level`
3. `bmi`
4. `hypertension` / `heart_disease`

Bu, klinik beklentilerle uyumlu — yaş ve kan şekeri inme riskinin başlıca göstergeleri.

## Kullanılan Kütüphaneler
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn  # SMOTE için

## Çalıştırma

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
jupyter notebook notebook52d489b3c9__1_.ipynb
```

Notebook Kaggle ortamında çalıştırıldığı için veri yolu `/kaggle/input/...` şeklinde. Lokalde çalıştıracaksanız ilk hücredeki `pd.read_csv(...)` yolunu kendi CSV konumunuza göre güncelleyin.

## Sonuç ve İyileştirme Önerileri

Mevcut model dengesiz veri probleminde recall'ı `0.16`'dan `0.40`'a çıkarmayı başardı, ancak precision hâlâ düşük (`0.13`). Gerçek bir tarama aracı için aşağıdaki adımlar denenebilir:

- XGBoost / LightGBM ile karşılaştırma
- Threshold tuning (varsayılan 0.5 yerine recall-precision dengesi için optimize)
- Daha fazla veri toplama, özellikle pozitif vaka sayısı
- Daha derin EDA: yaşa göre stratifikasyon, etkileşim terimleri
