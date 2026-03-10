# Graduation Project - Camera Based Customer Analytics

Bu proje, guvenlik kamerasi veya herhangi bir kameradan gelen goruntuler uzerinden musteri icin yas araligi, cinsiyet, gelis saati ve goruntu icinde gorulen urun benzeri nesneleri tespit ederek rapor uretmeyi hedefler. Cikis raporu CSV formatindadir ve analitik icin uygun sekilde gruplandirilir.

Bu README hem nasil calistirilacagini hem de kodun ne yaptigini adim adim aciklar.

## Amac ve Genel Akis

Sistem 3 ana adimdan olusur:

1. Kamera veya video kaynagini okur.
2. Yuz tespiti yapar, yuzden yas ve cinsiyet tahmini uretir.
3. Sahnedeki urun benzeri nesneleri (COCO etiketleri) tespit eder ve tum bilgileri raporlar.

## Kurulum

Gereksinimler:
- Python 3.9+
- Windows icin kamera izni

Kurulum adimlari:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Hizli Baslangic

1) Modeli egit:

```powershell
python src/training/train.py --epochs 5 --batch-size 64
```

Bu komut FairFace veriseti uzerinden yas ve cinsiyet siniflandirma modeli egitir. Egitim bittiginde model `models/age_gender_resnet18.pth` dosyasina kaydedilir.

2) Kamera demosu (rapor uretir):

```powershell
python src/inference/webcam_demo.py --model-path models/age_gender_resnet18.pth
```

Demo cikarken rapor `reports/events.csv` ve `reports/summary.csv` olarak olusur.

## Rapor Dosyalari

- `reports/events.csv`: Her bir musteri tespiti icin zaman damgasi, yas araligi, cinsiyet, urun etiketi ve guven skorlarini icerir.
- `reports/summary.csv`: Cinsiyet, yas araligi, urun ve saat bazinda ozet sayimlarini icerir.

## Kamera Demolari ve Parametreler

Varsayilan komut:

```powershell
python src/inference/webcam_demo.py
```

Onemli parametreler:
- `--camera-index`: Kamera indexi (varsayilan 0).
- `--video-path`: Kamera yerine video dosyasi kullanmak icin.
- `--model-path`: Egittiginiz model dosyasi.
- `--output-dir`: Rapor cikti klasoru.
- `--face-skip`: Yuz tespitini N karede bir calistirir. Performans icin artirabilirsiniz.
- `--product-skip`: Urun tespitini N karede bir calistirir. Performans icin artirabilirsiniz.
- `--product-score`: Urun tespit skoru esigi.
- `--no-products`: Urun tespitini devre disi birakir.
- `--no-display`: Ekranda pencere acmaz, sadece rapor uretir.

## Kod Yapisi (Dosya Dosya Aciklama)

- `src/config.py`: Tüm yol ve sabitleri tek noktada toplar. Model yolu, veri klasoru, etiket listeleri ve varsayilan ayarlar burada tutulur.
- `src/datasets/fairface_dataset.py`: FairFace CSV dosyasini okur ve PyTorch Dataset olarak goruntu + etiket dondurur.
- `src/utils/transforms.py`: Egitim ve dogrulama icin goruntu donusumlerini tanimlar.
- `src/models/age_gender_model.py`: ResNet18 tabanli iki cikisli (yas ve cinsiyet) siniflandirma modelini tanimlar.
- `src/utils/metrics.py`: Basit dogruluk hesaplamasi.
- `src/training/train.py`: Egitim dongusu. Modeli egitir, en iyi modeli kaydeder ve egitim gecmisini CSV olarak yazar.
- `src/utils/reporting.py`: Event listesine gore `events.csv` ve `summary.csv` raporlarini olusturur.
- `src/inference/webcam_demo.py`: Kamera/video akisini okur, yuz tespiti + yas/cinsiyet tahmini yapar, urun etiketi tespiti yapar ve raporlar.

## Performans ve Optimizasyon Notlari

- `--face-skip` ve `--product-skip` ile tespit araliklarini artirarak FPS artisi saglayabilirsiniz.
- Yuz tespiti OpenCV Haar Cascade ile yapilir. Hiza oncelik verir, ancak kalite dusuk isikta azalabilir.
- Urun tespiti COCO etiketleri uzerinden genel nesne algilama yapar. Gercek magazaya ozel urun tanimasi icin ek model veya POS entegrasyonu gerekir.
- Ilk calistirmada urun modeli agirliklarini indirmek gerekebilir. Offline ortamda `--no-products` secenegini kullanin.

## Etik ve Gizlilik Notu

Bu tur sistemler kisisel veri isler. Kurulum yaptiginiz ortamda KVKK/GDPR gibi mevzuatlara uymak, kullanicilari bilgilendirmek ve veri saklama politikalari belirlemek gerekir.
