# IST27 VCU Log & Performance Analyzer

IST27 VCU loglarını okuyup 100 Hz ızgaraya oturtan, sürüşü ve güç aktarma organını çizen Streamlit ekranı. Kaynak çerçeve CM7 `log_frame_t`: 96 bayt, CSV ayırıcı `;`.

Şema `config/vcu_signals_schema.yaml` içindedir. Ölçekler ve firmware boşlukları `vcu27_prep` kaynaklarına göre yazıldı: `log_frame.h`, `datalogger.c`, `app_tasks.c`, `can_ids.h`, `vehicle_config.h`, `vehicle_io.c`, `inverter_amk.c`.

## Çalıştırma

Python 3.10+ gerekir.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windows’ta `run_analyzer.bat` bağımlılıkları kurar, örnek CSV’leri üretir ve uygulamayı açar. Örnekleri elle üretmek için:

```bash
python generate_ist27_mock_logs.py
```

Loglar `logs_input/` altından okunur. Uzantılar: `.csv`, `.txt`, `.log`, `.bin`, `.parquet`. İki kayıt yan yana seçilebilir.

Repoda iki örnek vardır:

- `IST27_mock_clean_lap.csv` — temiz tur
- `IST27_mock_fault_session.csv` — APPS sapması, slip ve shutdown içeren oturum

Gerçek SD dosya adı biçimi: `/IST27/DD_MM_YY-hh_mm_ss.csv`.

## Ekranlar

| Sekme | Ne gösterir |
| --- | --- |
| Genel Bakış & Koçluk | Mesafe, Vmax, Pmax, 0–100, pilot ve VCU notları, darboğaz sırası, IsolationForest |
| 4 Motor & 2 İnverter | Tork/devir, slip, TV gecikmesi, ön ve arka inverter grupları |
| 600 V HV & 24 V LV | 80 kW limiti, erken kısma, iç direnç, LV brownout (sütun varsa) |
| Pilotaj & G-G | Gaz, fren, direksiyon, türetilmiş G-G, piezo FFT (sütun varsa) |
| Arıza / Shutdown | SDC, AMS, IMD, inverter ERROR/FAULT kenarı; olaydan 500 ms önce ve 200 ms sonra |

Güç, logdaki işaretsiz `pack_power_w` yerine `pack_v * pack_i` ile yeniden hesaplanır. Firmware regen anında gücü 0 yazar.

## Bilinen firmware boşlukları

Bu alanlar mimaride var, 96 baytlık çerçeveye yazılmıyor. Sütun CSV’ye eklenirse şema ölçekler; yoksa ilgili grafik bilgi notu gösterir.

- 24 V LV (`CAN 0x131` ve ADC) loglanmıyor
- APPS1/APPS2 ham ADC yok; sapma yalnızca `apps_status` bitlerinden okunur
- IMU’dan yalnızca yaw hızı var; G-G, boylamsal hız türevi ve `yaw * vx` ile türetilir
- Piezo, GPS, damper ve hücre voltajları yok
- AMK `error_info`, derating ve inverter soğutucu sıcaklığı yok
- Torque vectoring / traction control kancaları firmware’de nötr

## Düzen

```
app.py                         Streamlit ekranı
generate_ist27_mock_logs.py    Örnek CSV
config/vcu_signals_schema.yaml Sinyal ölçekleri, CAN ve eşikler
src/log_loader.py              CSV/binary okuma ve 100 Hz resample
src/vehicle_analytics.py       TV, slip, HV/LV, blackbox
src/driver_coaching.py         Pilot ve VCU metinleri
src/ml_engine.py               IsolationForest ve darboğaz sırası
src/plots.py                   Plotly figürleri
```
