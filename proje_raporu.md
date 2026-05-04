# WiFi Sinyalleriyle Temassız Vital Sign Monitorizasyonu
## Lisans Bitirme Projesi — Fizibilite ve Uygulama Raporu

**Tarih:** Mart 2026  
**Proje Türü:** Araştırma + Prototip  
**Zorluk Seviyesi:** Orta-İleri (ama yapılabilir)

---

## 1. Projenin Özeti

Bu proje, standart bir WiFi bağlantısından elde edilen **CSI (Channel State Information)** sinyallerini analiz ederek kişinin **kalp atışı ve nefes hızını** temassız olarak ölçmeyi hedeflemektedir. Kamera yok, özel donanım yok — sadece WiFi.

### Neden Bu Proje?

- Radar tabanlı yöntemler (UWB, FMCW) yüksek hassasiyet sunar ama özel donanım gerektirir
- WiFi her yerde var, ESP32 ~5$ maliyetle CSI verisi çekebiliyor
- Gizlilik: kamerasız, görüntüsüz
- Akademik literatürde aktif araştırma alanı — özgün katkı yapılabilir

---

## 2. Teknik Arka Plan

### CSI Nedir?

WiFi çerçeveleri gönderildiğinde, alıcı anten her subcarrier (alt taşıyıcı) için:
- **Genlik (amplitude):** Sinyalin gücü
- **Faz (phase):** Sinyalin gecikme miktarı

bilgisini ölçer. Bu bilgiye **Channel State Information (CSI)** denir.

```
[Verici ESP32] --WiFi--> [Ortam / İnsan Vücudu] ---> [Alıcı ESP32]
                                    ↑
                    Göğüs hareketi CSI'yi değiştirir:
                    - Nefes: 1-5 mm hareket, 0.1-0.5 Hz
                    - Kalp atışı: 0.1-0.5 mm hareket, 0.8-2.0 Hz
```

### Neden Çalışıyor?

IEEE 802.11n/ac protokolünde OFDM (Orthogonal Frequency Division Multiplexing) kullanılır. Her subcarrier bağımsız olarak ölçülür. 80 MHz bant genişliğinde 256 adet subcarrier vardır. Bu kadar ölçüm noktası küçük göğüs hareketlerini yakalamak için yeterlidir.

### İki Makalenin Katkısı

**Radar makalesi (2507.14195):**
- 2D+1D ResNet mimarisi vital sign için özel tasarlanmış
- Transfer learning → az veriyle yeni domaine geçiş
- Bu mimariyi WiFi'ye uyarlamak mümkün

**WiFi-DensePose (2301.00250):**
- CSI → spatial domain dönüşümü kanıtlandı
- Modality Translation Network konsepti kullanılabilir
- Vücut pozu + vital sign aynı pipeline'a entegre edilebilir

---

## 3. Proje Kapsamı (Lisans için Gerçekçi Hedefler)

### Minimum Hedef (Kesinlikle Yapılabilir)
- [ ] ESP32 ile CSI verisi topla
- [ ] Nefes hızını (0.1-0.5 Hz) başarıyla tespit et
- [ ] Basit FFT + peak detection ile çalışan sistem

### Orta Hedef (Makul Çaba ile Yapılabilir)
- [ ] Kalp atışını tespit et (zor ama mümkün)
- [ ] LSTM veya 1D-CNN modeli eğit
- [ ] Gerçek zamanlı görselleştirme arayüzü

### İdeal Hedef (İyi Olursa)
- [ ] Radar makalesindeki 2D+1D ResNet mimarisini WiFi'ye uyarla
- [ ] Transfer learning dene (varsa önceki CSI dataset ile)
- [ ] Vücut pozisyonu tahmini + vital sign birlikte

---

## 4. Donanım Gereksinimler

### Zorunlu (Toplam ~15-30$)

| Donanım | Adet | Fiyat | Açıklama |
|---------|------|-------|----------|
| ESP32 geliştirme kartı | 2 | ~5$/adet | Verici + Alıcı |
| USB kablo | 2 | ~1$/adet | Programlama için |
| Laptop/PC | 1 | Zaten var | Veri işleme |

> **Önemli:** ESP32 kullanılacaksa **Espressif ESP-IDF** ile programlanmalı. Arduino kütüphanesi CSI erişimi vermez.

### Opsiyonel (Kalite artırır)

| Donanım | Adet | Fiyat | Açıklama |
|---------|------|-------|----------|
| ESP32 (3 anten) | +1 | ~5$ | 3x3 MIMO = daha iyi sinyal |
| Raspberry Pi | 1 | ~35$ | Gerçek zamanlı işleme |
| Pulse oksimetre | 1 | ~10$ | Ground truth için |

### Donanım Yerleşimi

```
[ESP32 Verici]          [Kişi Oturur]          [ESP32 Alıcı]
     |___________________|_______________|
           ~1-2 metre          ~1-2 metre
```

Kişi verici ve alıcı arasında veya alıcının önünde oturmalı.

---

## 5. Yazılım Stack

### Veri Toplama

```
ESP32 (C/ESP-IDF)
    └── CSI callback → Serial/UDP → PC
```

Kullanılacak araç: [ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool)

### İşleme ve Model (Python)

```
Veri Akışı:
CSI Ham Veri → Ön İşleme → Özellik Çıkarma → Model → Tahmin

Kütüphaneler:
- numpy, scipy       : Sinyal işleme, FFT
- matplotlib         : Görselleştirme
- torch / tensorflow : Derin öğrenme modeli
- pyserial           : ESP32 serial okuma
- streamlit          : Basit arayüz (opsiyonel)
```

### Veri İşleme Pipeline

```python
# Basit nefes tespiti için örnek akış
1. CSI genlik al          → shape: (zaman, subcarrier)
2. Subcarrier seç         → en stabil 30 subcarrier
3. Bandpass filtre        → 0.1-0.5 Hz (nefes bandı)
4. FFT                    → dominant frekans = nefes hızı
5. Peak detection         → nefes/dakika hesapla
```

---

## 6. Veri Toplama Planı

### Deney Kurulumu

- **Kişi sayısı:** 2-3 kişi yeterli (lisans projesi için)
- **Session süresi:** Kişi başı 5 dk oturup nefes al
- **Pozisyonlar:** Oturma, yatma, ayakta (3 pozisyon)
- **Ground truth:** Pulse oksimetre ile eş zamanlı ölç

### Etiketleme Stratejisi

```
Zaman damgası → [CSI verisi | Gerçek nefes hızı | Gerçek kalp atışı]
```

### Kaç Veri Gerekir?

- Nefes tespiti (FFT tabanlı): Etiketli veri gerekmez, işaret işleme yeterli
- Kalp atışı (ML tabanlı): ~1000-2000 pencere (her 30 saniyelik pencere = 1 örnek)
- 3 kişi × 5 dk × 2 Hz örnekleme = yeterli

---

## 7. Model Mimarisi (Önerilen)

### Seçenek A — Basit (Güvenli, Önerilen)

```
CSI Genlik [T x N_subcarrier]
    → Bandpass Filtre
    → FFT
    → Peak Detection
    → Nefes hızı (bpm)
```

Makine öğrenmesi yok, işaret işleme. Başarı garantidir.

### Seçenek B — Orta (ML + Kalp Atışı)

```
CSI Ham [30s pencere x 52 subcarrier]
    → 1D-CNN (3 katman)
    → LSTM (64 unit)
    → Dense → bpm tahmini
```

Radar makalesindeki 2D+1D ResNet'in basitleştirilmiş versiyonu.

### Seçenek C — İdeal (Radar Makalesini Uyarla)

```
2D özellik: [subcarrier × zaman]  → 2D-CNN
1D özellik: [zaman serisi]         → 1D-CNN
    → Concat → FC → [nefes bpm, kalp bpm]
```

Literatürde yeni: WiFi üzerinde radar makalesinin mimarisini denemek özgün katkıdır.

---

## 8. Beklenen Sonuçlar ve Başarı Kriterleri

### Nefes Hızı

| Yöntem | Beklenen MAE | Zorluk |
|--------|-------------|--------|
| FFT tabanlı | 1-3 nefes/dk | Kolay |
| ML tabanlı | < 1 nefes/dk | Orta |

**Klinik kabul:** ±3 nefes/dakika

### Kalp Atışı

| Yöntem | Beklenen MAE | Zorluk |
|--------|-------------|--------|
| Basit FFT | 5-10 bpm | Zor |
| 1D-CNN/LSTM | 3-6 bpm | Zor |
| 2D+1D ResNet | 2-4 bpm | Çok zor |

**Radar makalesi referans:** 0.85 bpm (FMCW), 4.1 bpm (UWB transfer)  
**WiFi için gerçekçi hedef:** 4-6 bpm MAE

> Not: Kalp atışı WiFi ile zordur çünkü göğüs hareketi çok küçük (~0.1mm). Nefes hareketleri (1-5mm) çok daha kolay tespit edilir.

---

## 9. Zaman Çizelgesi (14 Haftalık Plan)

```
Hafta 1-2   : Literatür okuma, ESP32 kurulum, CSI veri toplama testi
Hafta 3-4   : Veri toplama kampanyası (3 kişi, çeşitli senaryolar)
Hafta 5-6   : Sinyal işleme (FFT, filtreleme) → nefes tespiti çalışır hale getir
Hafta 7-8   : Kalp atışı için ML modeli geliştir
Hafta 9-10  : Model eğitimi, hyperparameter ayarı
Hafta 11-12 : Gerçek zamanlı sistem + görselleştirme arayüzü
Hafta 13    : Test, hata analizi, iyileştirme
Hafta 14    : Rapor yazımı, sunum hazırlama
```

---

## 10. Riskler ve Çözümler

| Risk | Olasılık | Çözüm |
|------|----------|-------|
| Kalp atışı tespit edilemiyor | Orta | Nefes tespitine odaklan, kalp atışı bonus say |
| ESP32 CSI verisi gürültülü | Yüksek | Çok subcarrier ortalaması al, filtrele |
| Yeterli veri toplanamıyor | Düşük | Açık kaynak dataset kullan (ek kaynak) |
| Model overfitting | Orta | Kişiden bağımsız (leave-one-out) validation |
| Ortam değişince düşük performans | Yüksek | Tek odada çalış, sabitle, bunu limitation olarak yaz |

---

## 11. Açık Kaynak Kaynaklar (Sıfırdan Başlamanıza Gerek Yok)

### Kod

| Repo | Ne İşe Yarar |
|------|-------------|
| [ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool) | ESP32'den CSI çekme |
| [esp-csi (Espressif)](https://github.com/espressif/esp-csi) | Resmi Espressif CSI örneği |
| [csi_hr](https://github.com/nickbild/csi_hr) | WiFi ile kalp atışı (LSTM) |
| [WiFi-CSI-MiningTool](https://github.com/AlbanyArmenta0711/WiFi-CSI-MiningTool) | CSI işleme scripti + dataset |
| [RuView](https://github.com/ruvnet/RuView) | Tam pipeline (poz + vital sign) |

### Dataset (Veri toplamaya vaktiniz yoksa)

| Dataset | İçerik |
|---------|--------|
| WiFi-CSI-MiningTool dataset | 17 kişi, nefes + kalp atışı |
| [Awesome-WiFi-CSI-Sensing](https://github.com/Marsrocky/Awesome-WiFi-CSI-Sensing) | Derlenmiş kaynak listesi |

---

## 12. Özgünlük: Ne Ekleyebiliriz?

Sadece mevcut çalışmaları tekrar yapmak yerine küçük ama özgün katkılar:

1. **Radar → WiFi Transfer Learning:** Radar makalesinin 2D+1D ResNet mimarisini WiFi CSI'ye uyarlamak (literatürde yok)
2. **Türkçe dil ortamı verisi:** Lokal veri toplama + raporlama
3. **Gerçek zamanlı sistem:** Çoğu makale offline; gerçek zamanlı çalışan prototip değerli
4. **Karşılaştırmalı analiz:** FFT vs CNN vs LSTM — hangisi WiFi'de daha iyi?

---

## 13. Sonuç: Yapılabilir mi?

### Evet, yapılabilir.

**Nefes hızı:** Kesinlikle yapılabilir. Sinyal işleme ile bir haftada çalışır.  
**Kalp atışı:** Zor ama makul. ML modeli gerekiyor, 4-6 bpm doğruluk hedeflenebilir.  
**Vücut pozu:** Kapsam dışı bırakın, zamanı olmaz.

**Proje değeri:** Bu konu aktif araştırma alanı, lisans için güçlü bir proje.  
**Bütçe:** 30$ donanım, geri kalanı yazılım.  
**Takım:** 2-3 kişi idealdir.

---

## Kaynaklar

- Radar makalesi: https://arxiv.org/html/2507.14195v1
- WiFi-DensePose: https://arxiv.org/abs/2301.00250
- Person-in-WiFi: https://arxiv.org/abs/1904.00276
- ESP32-CSI-Tool: https://github.com/StevenMHernandez/ESP32-CSI-Tool
- csi_hr (kalp atışı): https://github.com/nickbild/csi_hr
- RuView (tam sistem): https://github.com/ruvnet/RuView
- WiFi CSI Survey: https://pmc.ncbi.nlm.nih.gov/articles/PMC9375645/

