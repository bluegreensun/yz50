# YZ50 — Week 1: Neural Networks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bluegreensun/yz50-week1-neural-networks/blob/main/YZ50_Week1_Neural_Networks.ipynb)

Tek bir nöronun ve küçük bir katmanın sıfırdan, hazır bir framework kullanmadan (saf Python + matplotlib) nasıl hesaplandığını adım adım gösteren notebook.

## İçerik

`YZ50_Week1_Neural_Networks.ipynb` beş adımdan oluşur:

1. **Tek nöron** — 3 girdi (`x1, x2, x3`), 3 ağırlık ve bir bias ile `z = x·w + b` hesabı.
2. **3 nöronlu katman** — `weights` matrisi (3×3) ve `biases` listesi ile `layer_outputs`.
3. **Loss** — hedef değerlere (`targets`) göre Mean Squared Error (MSE).
4. **Loss eğrisi** — `weights[0][0]` -60 ile 40 arasında değişirken MSE'nin nasıl değiştiğinin grafiği.
5. **Gradient descent** — sayısal türev (finite difference, `h = 0.0001`) ile tek bir ağırlığın 50 adımda güncellenmesi; loss'un ve `w`'nin adım adım gidişi.

## Çalıştırma

- **Google Colab:** Notebook'u Colab'da açıp hücreleri sırayla çalıştırmak yeterli.
- **Yerel:**

  ```bash
  pip install -r requirements.txt
  jupyter notebook YZ50_Week1_Neural_Networks.ipynb
  ```

İlk iki hücre `input()` ile kullanıcıdan sayı bekler; 4. hücreden itibaren değerler sabitlenmiştir (`x1, x2, x3 = 1.0, 2.0, 3.0`).

## Hafta 2 — Backpropagation (micrograd)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bluegreensun/yz50-week1-neural-networks/blob/main/hafta-2/YZ50_Week2_Backpropagation.ipynb)

`hafta-2/YZ50_Week2_Backpropagation.ipynb` — `Value` sınıfı ve computation graph, elle gradient doldurma, `backward()` ile otomatik geri yayılım, `tanh`'ın `exp`/`pow`/bölme ile parçalanması, sayısal türev + PyTorch ile üçlü doğrulama (Görev 1–4).
