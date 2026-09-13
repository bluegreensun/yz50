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

## Hafta 4 — MLP Dil Modeli (makemore part 2)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bluegreensun/yz50/blob/main/hafta-4/YZ50_Week4_MLP_Language_Model.ipynb)

`hafta-4/YZ50_Week4_MLP_Language_Model.ipynb` — Bengio 2003 tarzı çok katmanlı algılayıcı (MLP) ile karakter düzeyinde dil modeli. Hafta 3'ün bigram modelinin aksine 3 harflik bağlama (`block_size = 3`) bakılıyor ve karakterler öğrenilebilir embedding'lerle temsil ediliyor (Görev 1–5 ve 7):

1. **Veri seti ve embedding** — `names.txt` otomatik indirilir; kayan bağlam penceresiyle `X` (N, 3) / `Y` (N,) kurulumu, `C` embedding tablosu ve `C[X]` ile çok boyutlu indeksleme (one-hot çarpımıyla aynı sonuç).
2. **Forward pass ve loss** — `.view` ile düzleştirme (`cat`'ten neden ucuz olduğu), `tanh` gizli katman, logits; loss'un elle hesabı ile `F.cross_entropy`'nin aynı sonucu verdiğinin doğrulanması ve softmax'ta `exp` taşmasının max çıkarılarak engellenmesi.
3. **Eğitim döngüsü** — %80/%10/%10 train/dev/test bölmesi, tek minibatch'i kasten overfit etme, learning rate taraması (log-scale 0.001–1), 40.000 adım minibatch eğitimi + son %25'te lr decay. Sonuç: train 2.295 / dev 2.529 — bigramdan iyi ama underfitting.
4. **Modeli büyütme** — dört konfigürasyon (emb 2/10 × hidden 100/300) karşılaştırması; en iyisi emb=10, hidden=100 ile dev 2.464 (bigram 2.591). Ayrıca 2 boyutlu karakter embedding'lerinin grafiği ve iki modelden üretilen isimlerin yan yana kıyası.
5. **Initialization** — kötü init'in iki ayrı problemi: geniş logits (başlangıç loss'u 18.3, beklenen log(27)=3.3) ve doymuş `tanh` (%61.9). Sadece çıkış katmanını düzeltmek ilkini çözüyor, ikincisini değil; Kaiming ölçeği (`gain/√fan_in`, tanh için gain=5/3) doymayı %10.8'e indirip dev loss'u 2.453'ten 2.373'e çekiyor.
6. **Türkçe isimler** (Görev 7) — `turkce_isim_clean.txt` (13.865 isim, ç ğ ı ö ş ü dahil 30 karakterlik alfabe) ile aynı pipeline. Dosya alfabetik sıralı olduğu için bölmeden önce karıştırılıyor. MLP dev 2.151, bigram dev 2.491.

Veri dosyası (`turkce_isim_clean.txt`) klasörün içinde; Colab'da açınca sol panelden yüklemek gerekir (`names.txt` notebook tarafından indirilir). Yerelde:

```bash
cd hafta-4
pip install -r requirements.txt
jupyter notebook YZ50_Week4_MLP_Language_Model.ipynb
```
