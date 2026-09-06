# YZ50

Bu repo, YZ50 programındaki haftalık çalışmalarımı içerir; her haftanın teslimi kendi klasöründedir.

## Hafta 1 — Neural Networks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bluegreensun/yz50/blob/main/hafta-1/YZ50_Week1_Neural_Networks.ipynb)

Tek bir nöronun ve küçük bir katmanın sıfırdan, hazır bir framework kullanmadan (saf Python + matplotlib) nasıl hesaplandığını adım adım gösteren notebook.

`hafta-1/YZ50_Week1_Neural_Networks.ipynb` beş adımdan oluşur:

1. **Tek nöron** — 3 girdi (`x1, x2, x3`), 3 ağırlık ve bir bias ile `z = x·w + b` hesabı.
2. **3 nöronlu katman** — `weights` matrisi (3×3) ve `biases` listesi ile `layer_outputs`.
3. **Loss** — hedef değerlere (`targets`) göre Mean Squared Error (MSE).
4. **Loss eğrisi** — `weights[0][0]` -60 ile 40 arasında değişirken MSE'nin nasıl değiştiğinin grafiği.
5. **Gradient descent** — sayısal türev (finite difference, `h = 0.0001`) ile tek bir ağırlığın 50 adımda güncellenmesi; loss'un ve `w`'nin adım adım gidişi.

**Çalıştırma** — Colab'da açıp hücreleri sırayla çalıştırmak yeterli. Yerelde:

```bash
cd hafta-1
pip install -r requirements.txt
jupyter notebook YZ50_Week1_Neural_Networks.ipynb
```

İlk iki hücre `input()` ile kullanıcıdan sayı bekler; 4. hücreden itibaren değerler sabitlenmiştir (`x1, x2, x3 = 1.0, 2.0, 3.0`).

## Hafta 2 — Backpropagation (micrograd)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bluegreensun/yz50/blob/main/hafta-2/YZ50_Week2_Backpropagation.ipynb)

`hafta-2/YZ50_Week2_Backpropagation.ipynb` — `Value` sınıfı ve computation graph, elle gradient doldurma, `backward()` ile otomatik geri yayılım, `tanh`'ın `exp`/`pow`/bölme ile parçalanması, sayısal türev + PyTorch ile üçlü doğrulama (Görev 1–4).

## Hafta 3 — Bigram Language Model (makemore)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bluegreensun/yz50/blob/main/hafta-3/YZ50_Week3_Bigram_Language_Model.ipynb)

`hafta-3/YZ50_Week3_Bigram_Language_Model.ipynb` — karakter düzeyinde bigram dil modeli, önce sayım tablosuyla sonra tek katmanlı bir neural net ile (Task 1–5):

1. **Bigram sayımı** — `names.txt` (32.033 İngilizce isim), `.` başlangıç/bitiş işareti, 27×27 `N` sayım matrisi ve ısı haritası.
2. **Olasılık tablosu + sampling** — satır normalizasyonu (`keepdim=True` broadcasting tuzağı), `torch.multinomial` ile isim üretme.
3. **Loss** — log likelihood, negative log likelihood ve ortalama NLL (2.4541); olasılıkları doğrudan çarpınca underflow, `+1` smoothing.
4. **Neural net karşılığı** — one-hot girdi, 27×27 `W`, softmax, NLL + regularization, gradient descent; `softmax(W)`'nin count modelindeki `P` tablosuna yakınsadığının gösterimi.
5. **Türkçe isimler** — `turkce_isim_clean.csv` (13.865 isim, ç ğ ı ö ş ü dahil 30 karakterlik alfabe) ile aynı pipeline: count modeli, neural net ve Türkçe isim üretimi.

Veri dosyaları (`names.txt`, `turkce_isim_clean.csv`) klasörün içinde; Colab'da açınca ikisini sol panelden yüklemek gerekir. Yerelde:

```bash
cd hafta-3
pip install -r requirements.txt
jupyter notebook YZ50_Week3_Bigram_Language_Model.ipynb
```
