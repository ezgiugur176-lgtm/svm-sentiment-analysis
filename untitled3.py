#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 15:35:44 2025

@author: ezgiugur
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

dosya_adi = 'product data.xlsx' 
df = pd.read_excel(dosya_adi)

metin_sutunu = 'text'
etiket_sutunu = 'rating'

df.dropna(subset=[metin_sutunu, etiket_sutunu], inplace=True)

yorumlar = df[metin_sutunu]
etiketler = df[etiket_sutunu]

print(f"Yüklenen ve Temizlenen Yorum Sayısı: {len(df)}")

etiketler_ikili = etiketler.apply(lambda x: 1 if x >= 4 else 0)

vectorizer = TfidfVectorizer(
    lowercase=True,
    max_features=1000,
    stop_words='english'
)

X = vectorizer.fit_transform(yorumlar)

print(f"TF-IDF Matrisinin Boyutu (Özellik Kümesi): {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, etiketler_ikili, 
                                                    test_size=0.2, 
                                                    random_state=42)
svm_model = SVC(kernel='linear', C=1.0, random_state=42)

print("SVM Modeli Eğitiliyor...")
svm_model.fit(X_train, y_train)
print("Eğitim Tamamlandı.")

y_pred = svm_model.predict(X_test)

print("\n### Sınıflandırma Raporu (SVM Performansı) ###")
print(classification_report(y_test, y_pred, zero_division=0))


kategori_sutunu = 'Category'
X_tum = vectorizer.transform(yorumlar) 

df['Tahmin_Etiket'] = svm_model.predict(X_tum) 


kategori_analizi = df.groupby(kategori_sutunu)['Tahmin_Etiket'].agg(
    Toplam_Yorum='count',
 
    Pozitif_Yuzde=lambda x: (x.sum() / x.count()) * 100
).reset_index()

print("\n### Kategori Bazında Pozitiflik Yüzdesi (Sonuç) ###")
print(kategori_analizi.sort_values(by='Pozitif_Yuzde', ascending=False))


