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

## Hafta 6 — WaveNet (makemore part 5)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bluegreensun/yz50/blob/main/hafta-6/YZ50_Week6_WaveNet_Turkce.ipynb)

`hafta-6/YZ50_Week6_WaveNet_Turkce.ipynb` — Hafta 4'ün düz MLP'si yerine, bağlamı tek adımda değil kademeli olarak birleştiren WaveNet tarzı hiyerarşik bir ağ; Türkçe isim dataseti üzerinde (Görev 6):

1. **Veri seti** — `turkce_isim_clean.txt` (13.865 isim, ç ğ ı ö ş ü dahil 30 karakterlik alfabe), `block_size = 8` ile kayan bağlam penceresi ve Hafta 4 ile aynı %80/%10/%10 bölmesi (train 79.430 / dev 10.320 / test 10.123 örnek).
2. **Katman sınıfları** — PyTorch'un `nn.Module`'üne karşılık gelen kendi sınıfları: `Embedding`, `Flatten`, `FlattenConsecutive`, `Linear`, `BatchNorm1d` (2D `(B, C)` ve 3D `(B, T, C)` girdiyi birlikte destekleyecek şekilde), `Tanh` ve katmanları sırayla çağıran `Sequential`.
3. **Hiyerarşik mimari** — `FlattenConsecutive(2)` ile bağlamın ikişer ikişer eşlenmesi: 8 → 4 → 2 → 1. Her adımda `(B, T, C) -> (B, T/2, C*2)` ve ardından `Linear + BatchNorm + tanh`; emb=10, hidden=100. Shape akışı hücre çıktısında adım adım yazdırılıyor.
4. **Eğitim** — 40.000 adım, batch 32, son %25'te lr 0.1 → 0.01 decay; loss eğrisi 1000 adımlık ortalamalarla çiziliyor. Eğitim sonunda BatchNorm katmanları `training = False` ile running istatistiklere geçiyor.
5. **Karşılaştırma** — aynı kelime listesi ve aynı bölme üzerinde Hafta 4 Türkçe MLP baseline'ı (context=3) yeniden eğitiliyor. Dev loss: MLP 2.3815 → WaveNet 2.2717 (0.1098 iyileşme); daha uzun bağlam ve kademeli birleştirme validation performansını iyileştirdi.
6. **Örnekleme** — modelden üretilen Türkçe isimler (`cantürk`, `efeki`, `akaygun`, `gülsan`, `kerkaya`, `hürhalit`, `sertan` …).

Veri dosyası (`turkce_isim_clean.txt`) klasörün içinde; Colab'da açınca sol panelden yüklemek gerekir. Yerelde:

```bash
cd hafta-6
pip install -r requirements.txt
jupyter notebook YZ50_Week6_WaveNet_Turkce.ipynb
```

## Hafta 7 — Self-Attention (Let's build GPT)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bluegreensun/yz50/blob/main/hafta-7/YZ50_Week7_GPT_Self_Attention.ipynb)

`hafta-7/YZ50_Week7_GPT_Self_Attention.ipynb` — isim verisinden tek parça metne geçiş: Tiny Shakespeare üzerinde karakter düzeyinde bigram tabanı, self-attention'ın arkasındaki matris hilesi ve tek bir self-attention head'li model (Görev 1–4):

1. **Bigram tabanı** — Tiny Shakespeare (1.115.394 karakter, 65 karakterlik vocab) otomatik indirilir; karakter tokenizer, karıştırmadan %90/%10 train/val bölmesi, `(B, T)` chunk'lardan batch. `nn.Embedding(65, 65)` bigram modeli 3000 adımda val 2.4907'ye iniyor (sayım modeli 2.4819); sonuç Görev 4 için `bigram_baseline.json`'a kaydediliyor.
2. **Matris hilesi** — her pozisyonun geçmişinin ortalaması üç yoldan: `for` döngüsü, `torch.tril` ile matris çarpımı ve `masked_fill(-inf)` + softmax. Üçünün aynı sonucu verdiği (float32 toleransı ile) ve matris çarpımının `B=32, T=256, C=64`'te döngüden ~44x hızlı olduğu gösteriliyor; rastgele skorlarla softmax'ın ağırlıklı ortalamaya dönüşmesi.
3. **Tek self-attention head** — token + pozisyon embedding'i, query/key/value, `(B, T, T)` ağırlık matrisinin bir dizi üzerinde satır satır okunması; pozisyon embedding'i olmadan aynı karakterin iki farklı yerdeki key'inin ayırt edilemediği ve `√head_size`'a bölmenin skor varyansını 1'e çektiği.
4. **Eğitim ve karşılaştırma** — tek head'li model (7.553 parametre) ile head'siz kontrol modeli aynı ayarlarla eğitiliyor. Val loss: bigram 2.4907, head'siz 2.4963, tek head 2.4047 (%3.5 düşüş, perplexity 12.1 → 11.1). Pozisyon bazında kazancın tamamı bağlamın olduğu t=1..7'den geliyor; üretimde girdinin neden son `block_size` karaktere kırpıldığı.

Veri dosyası notebook tarafından indirilir. Yerelde:

```bash
cd hafta-7
pip install -r requirements.txt
jupyter notebook YZ50_Week7_GPT_Self_Attention.ipynb
```

## Hafta 8 — Multi-Head Attention ve Transformer (Let's build GPT, devam)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bluegreensun/yz50/blob/main/hafta-8/YZ50_Week8_GPT_Transformer.ipynb)

`hafta-8/YZ50_Week8_GPT_Transformer.ipynb` — Hafta 7'nin tek head'li modelinden adım adım küçük bir GPT'ye: multi-head attention, feedforward, residual bağlantılar ve LayerNorm; ardından modeli büyütme ve Türkçe metin üzerinde eğitim (Görev 1–4):

1. **Multi-head attention** — aynı parametre bütçesiyle (7.553) tek 1×32 head yerine 4×8 head. Önce Hafta 7'nin tek head sonucu birebir yeniden üretiliyor (val 2.4047); 4 head ile val 2.2693 (0.1354 düşüş, perplexity 11.08 → 9.67). Pozisyon bazında kazanç bağlamın olduğu t=1..7'den geliyor (her pozisyonda ~0.13–0.17), t=0'da fark yok.
2. **Transformer bloğu** — bileşenler tek tek eklenerek val loss: bigram 2.4907 → tek head 2.4047 → multi-head 2.2693 → + feedforward 2.2358 → 3 blok + projeksiyon + residual (ffwd 4x) 2.0666 → + LayerNorm 2.0568. Residual olmadan 3 bloğu üst üste koymanın val loss'u kötüleştirdiği (2.3319 / 2.3828) ablation satırlarıyla gösteriliyor. Ayrıca LayerNorm ile BatchNorm1d karşılaştırması: hangi eksende normalize ettikleri, bir örneğin çıktısının batch'teki diğer örneklere bağlılığı, train/eval farkı, batch=1'de NaN ve `(B, T, C)` girdide BatchNorm'un gelecekteki pozisyonlardan bilgi sızdırması; kendi `LayerNorm1d` sınıfının `nn.LayerNorm` ile aynı sonucu verdiği.
3. **Modeli büyütme** — n_embd 128, 4 head (head_size 32), 4 katman, block_size 64, dropout 0.1: 816.705 parametre, CPU'da 4000 adım (~25 dk). Val loss 1.6366; üretilen metin karakter adları ve satır yapısıyla Shakespeare formatına belirgin şekilde yaklaşıyor.
4. **Türkçe metin** — aynı mimari `turkish.txt` (994.750 karakter, ç ğ ı ö ş ü dahil 72 karakterlik vocab) üzerinde: 818.504 parametre, val loss 0.2081. Val satırlarının %71.3'ü train'de birebir geçtiği için bu loss, Shakespeare'in 1.6366'sıyla doğrudan kıyaslanamaz.

Görev 1–3'ün verisi (Tiny Shakespeare) notebook tarafından indirilir; Görev 4 için `turkish.txt` dosyasını Colab'da sol panelden yüklemek gerekir. Yerelde:

```bash
cd hafta-8
pip install -r requirements.txt
jupyter notebook YZ50_Week8_GPT_Transformer.ipynb
```
