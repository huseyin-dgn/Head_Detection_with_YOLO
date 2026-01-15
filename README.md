# Head Detection with YOLO

Bu depo, **Ultralytics YOLO** tabanlı bir **head detection (kafa tespiti)** sistemi geliştirmek amacıyla oluşturulmuştur. Çalışma; veri hazırlama, model eğitimi, değerlendirme (metrik ve sayım bazlı) ve sonuçların görselleştirilmesi aşamalarını içeren **uçtan uca (end‑to‑end)** bir yapıya sahiptir.

Repo, akademik raporlama ve endüstriyel PoC (proof‑of‑concept) senaryolarına uygun olacak şekilde **aşamalı (stage‑based)** ve düzenli biçimde yapılandırılmıştır.

---

## Temel Özellikler

* Ultralytics YOLO (YOLOv8) tabanlı head detection modeli
* Aşamalı deney yapısı (training → evaluation → inference)
* Conf / IoU eşiklerine bağlı performans karşılaştırmaları
* Ground Truth vs Prediction sayım analizi
* Görsel sonuçlar ve model logları ile şeffaf deney takibi

---

## Proje Klasör Yapısı

```text
Head_Detection/
├── first_stage/        # Veri hazırlama ve ön işlemler
├── second_stage/       # Model eğitimi (training)
├── third_stage/        # Nihai değerlendirme ve sonuçlar
│   ├── result.png      # Model çıktılarının görsel özeti
│   ├── *.log           # Eğitim / değerlendirme log dosyaları
│   └── outputs/        # Tahmin sonuçları (txt / görsel)
├── dataset.yaml        # YOLO veri yapılandırması
└── README.md
```

> **Not:** `third_stage` klasörü, çalışmanın en kritik çıktılarının bulunduğu aşamadır.

---

## Third Stage – Nihai Sonuçlar

### 🔹 Görsel Sonuç

Aşağıdaki görsel, modelin head detection performansını ve sayım davranışını özetlemektedir:

```text
third_stage/result.png
```

Bu görselde:

* Tespit edilen kafalar
* Conf ve IoU eşiklerine göre dağılım
* Görsel doğrulama (qualitative evaluation)

açık biçimde gözlemlenebilir.

---

### 🔹 Model ve Değerlendirme Logları

Model eğitimi ve değerlendirme sürecinde üretilen log dosyaları yine aynı klasörde yer almaktadır:

```text
third_stage/
├── train.log
├── eval.log
├── metrics.log
```

Bu loglar üzerinden:

* Epoch bazlı kayıp (loss) değişimleri
* Precision / Recall / mAP trendleri
* Sayım bazlı hata metrikleri (MAE, RMSE vb.)

izlenebilir ve deneyler **tekrar üretilebilir** hale gelir.

---

## Kullanılan Teknolojiler

* **Python 3.9+**
* **Ultralytics YOLOv8**
* PyTorch
* NumPy / Pandas
* OpenCV

---

##

```bash
```

---

## Eğitim (Training)

```bash
yolo detect train \
  data=dataset.yaml \
  model=yolov8n.pt \
  imgsz=960 \
  epochs=80 \
  batch=8
```

---

## Değerlendirme (Evaluation)

Model çıktıları, **conf / IoU eşiklerine göre** değerlendirilmiş ve sayım hataları analiz edilmiştir.

Öne çıkan metrikler:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Square Error)
* Toplam GT vs Prediction farkı

Detaylı sonuçlar `third_stage` altında bulunmaktadır.

---

## Amaç ve Kullanım Alanları

Bu proje özellikle:

* Kalabalık sahnelerde insan sayımı
* CCTV / güvenlik sistemleri
* Akıllı şehir uygulamaları
* Yoğunluk analizi

gibi senaryolara yönelik bir **başlangıç referansı** olarak tasarlanmıştır.

---

## Lisans

Bu proje akademik ve kişisel kullanım için açıktır. Ticari kullanım öncesi ilgili lisans koşullarını gözden geçiriniz.

---

## İletişim

Geliştirici: **Hüseyin DGN**
GitHub: [https://github.com/huseyin-dgn](https://github.com/huseyin-dgn)
