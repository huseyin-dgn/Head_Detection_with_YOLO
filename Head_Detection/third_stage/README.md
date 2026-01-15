# Head Counting – YOLOv8 Fine-Tuning (Final Model)

Bu çalışma, **SCUT-HEAD** tabanlı yeniden dengelenmiş (rebalanced) bir veri seti üzerinde
YOLOv8 tabanlı bir **head counting (insan sayma)** modelinin fine-tune edilmesini ve
sayım odaklı performansının optimize edilmesini kapsamaktadır.

---

## 1. Kullanılan Model

- **Backbone:** YOLOv8n (Ultralytics)
- **Task:** Object Detection (single class – head)
- **Framework:** Ultralytics YOLO v8.3.x
- **Device:** NVIDIA RTX 3060 Laptop GPU (6 GB VRAM)
- **Precision:** AMP (Automatic Mixed Precision)

---

## 2. Eğitim Konfigürasyonu (Final)

| Parametre    | Değer                           |
| ------------ | ------------------------------- |
| Image Size   | 896 × 896                       |
| Epochs       | 30                              |
| Batch Size   | 4                               |
| Optimizer    | AdamW                           |
| Initial LR   | 0.002                           |
| Weight Decay | 0.0005                          |
| Mosaic       | Enabled (closed after epoch 25) |
| Patience     | 10                              |
| Augmentation | RandAugment (default YOLOv8)    |

---

## 3. Eğitim Sonuçları (Detection)

Final model doğrulama (val) sonuçları:

- **mAP@50:** 0.967
- **mAP@50–95:** 0.532
- **Precision:** 0.942
- **Recall:** 0.937

Model, eğitim boyunca stabil bir şekilde yakınsamış ve overfitting gözlemlenmemiştir.

---

## 4. Sayım (Counting) Performansı

Sayım değerlendirmesi, model çıktılarının farklı **confidence (conf)** ve **IoU**
eşiklerinde post-process edilmesiyle yapılmıştır.

### En İyi Konfigürasyon

- **Confidence:** 0.34
- **IoU:** 0.45

### Sayım Metrikleri

| Metric             | Değer      |
| ------------------ | ---------- |
| N_images           | 350        |
| GT_total           | 9159       |
| Pred_total         | 9166       |
| Abs_total_diff     | **7**      |
| MAE                | **1.2886** |
| RMSE               | **2.6560** |
| MAPE (ignore gt=0) | **0.0576** |
| Bias (Pred − GT)   | **+7**     |

Bu sonuçlar, veri seti ölçeği dikkate alındığında **yüksek doğruluklu ve dengeli**
bir sayım performansına işaret etmektedir.

---

## 5. Model Dosyası

Final model ağırlıkları:

```text
C:\Users\hdgn5\OneDrive\Masaüstü\Head Detection\second stage\B_v8n_finetune_img896_ep30_bs2_AdamW_lr0022\weights\best.pt
```
