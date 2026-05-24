# İnme (Stroke) Tahmini — Keşifçi Veri Analizinden (EDA) Makine Öğrenmesine

Bu proje, [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) veri seti üzerinde gerçekleştirilmiş uçtan uca bir makine öğrenmesi çalışmasıdır. Projenin ana odağı, körü körüne yüksek doğruluk (Accuracy) skoru peşinde koşmak değil; veri setindeki **ciddi sınıf dengesizliğini (~%5 pozitif inme oranı)** yönetmek ve bir klinisyenin veya uzmanın gerçekten işine yarayabilecek tahminler üretebilmektir.

## 🛠️ Proje İçeriği

1. **Klinik EDA (Keşifçi Veri Analizi)** — Özelliklerin dağılımları, Boxplot (Kutu Grafiği) ile inme risk faktörlerinin analizi ve sayısal/kategorik değişkenlerin inme durumu üzerindeki etkilerinin incelenmesi.
2. **Sınıf Dengesizliği ile Mücadele** — Verideki dengesizliği çözmek adına sadece Eğitim (Train) verisine uygulanan **SMOTE (Synthetic Minority Over-sampling Technique)** yöntemi.
3. **Model Eğitimi ve Optimizasyon Kıymeti** — Rastgele Orman (**Random Forest Classifier**) algoritmasının eğitimi, hiperparametre optimizasyonu (GridSearchCV) denemesi ve aşırı öğrenme (overfitting) analizleri.
4. **Metrik Odaklı Değerlendirme** — Sağlık sektörüne uygun olarak Accuracy yerine **Recall (Duyarlılık)**, Precision (Kesinlik) ve Hata Matrisi (Confusion Matrix) odaklı final modeli seçimi.

## 📈 Önemli Bulgular ve Çıkarımlar

- **Yaş En Güçlü Belirteçtir:** Hem yapılan EDA analizlerinde (inme geçirenlerin yaş medyanının 70+ olması) hem de eğitilen Random Forest modelinin Özellik Önem Derecesinde (Feature Importance) **Yaş (Age)** açık ara en dominant risk faktörü olarak çıkmıştır.
- **Sınıf Dengesizliği Yönetimi Şarttır:** Standart Random Forest modeli, verideki sağlıklı kişilerin ezici çoğunluğundan dolayı inme vakalarının sadece **%16'sını** yakalayabilmiştir (Recall: 0.16). SMOTE entegrasyonu ile bu oran **%40'a (Recall: 0.40) çıkarılmıştır.**
- **Metrik Dengesi (Trade-off):** Gerçek inme vakalarını daha çok yakalamaya başladığımızda (Recall artışı), modelimiz doğal olarak daha fazla kişiye "inme riski var" diyerek yanlış alarm (Precision düşüşü) vermeye başlamıştır. Sağlık projelerinde yanlış negatifler (hastayı kaçırmak), yanlış pozitiflerden (yanlış alarm) çok daha maliyetli olduğu için bu durum kabul edilmiştir.

## 💻 Projeyi Çalıştırma

### Kaggle Üzerinde
1. Veri setinin Kaggle sayfasında yeni bir notebook açın.
2. İndirdiğiniz `.ipynb` uzantılı analiz dosyasını yükleyin.
3. Tüm hücreleri sırayla çalıştırın. `zipfile` ve yol kontrolleri otomatik olarak veriyi hafızaya alacaktır.


## 🚀 Kullanılan Teknolojiler
* Python 3
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-Learn
* Imbalanced-Learn (SMOTE)
