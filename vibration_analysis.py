import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, spectrogram, find_peaks
from scipy.stats import skew, kurtosis
import pandas as pd


# ---------- Настройки фильтра ----------
def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='high', analog=False)


def highpass_filter(data, cutoff, fs, order=4):
    b, a = butter_highpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)


# ---------- Расчёт метрик ----------
def vibration_metrics(x):
    return {
        "RMS": np.sqrt(np.mean(x**2)),
        "Peak-to-Peak": np.ptp(x),
        "Crest Factor": np.max(np.abs(x)) / np.sqrt(np.mean(x**2)),
        "Skewness": skew(x),
        "Kurtosis": kurtosis(x)
    }


# ---------- Основная функция анализа ----------
def analyze_vibration(signal, fs, rpm=None, show_plots=True):
    """
    signal: массив сигналов (виброускорение)
    fs: частота дискретизации, Гц
    rpm: скорость вращения (об/мин)
    """
    t = np.arange(len(signal)) / fs

    # Фильтрация
    x_filt = highpass_filter(signal, 5, fs)
    
    # FFT
    n = len(x_filt)
    freq = np.fft.rfftfreq(n, 1/fs)
    fft_mag = np.abs(np.fft.rfft(x_filt)) / n

    # Огибающая (Hilbert)
    envelope = np.abs(hilbert(x_filt))

    # Основные метрики
    metrics = vibration_metrics(x_filt)

    # Поиск пиков
    peaks, _ = find_peaks(fft_mag, height=np.max(fft_mag)*0.3)
    peak_freqs = freq[peaks]
    peak_amps = fft_mag[peaks]

    # Простая эвристическая диагностика
    diagnosis = {}
    if rpm:
        f1 = rpm / 60.0
        diagnosis["Imbalance"] = any(np.isclose(peak_freqs, f1, atol=1))
        diagnosis["Misalignment/Rubbing"] = any(np.isclose(peak_freqs, 2*f1, atol=1))
        diagnosis["Bearing defect (broadband)"] = np.sum(fft_mag[freq > 200]) > np.sum(fft_mag[freq < 200]) * 0.5

    # ---------- Визуализация ----------
    if show_plots:
        plt.figure(figsize=(14, 8))

        plt.subplot(3, 1, 1)
        plt.plot(t, x_filt)
        plt.title("Временной сигнал (виброускорение)")
        plt.xlabel("Время, с")
        plt.ylabel("Амплитуда")

        plt.subplot(3, 1, 2)
        plt.plot(freq, fft_mag)
        plt.title("Амплитудный спектр")
        plt.xlabel("Частота, Гц")
        plt.ylabel("Амплитуда")
        plt.grid(True)

        plt.subplot(3, 1, 3)
        f, t_spec, Sxx = spectrogram(x_filt, fs=fs, nperseg=1024, noverlap=512)
        plt.pcolormesh(t_spec, f, 10*np.log10(Sxx + 1e-20), shading='gouraud')
        plt.title("Спектрограмма (энергия по времени)")
        plt.xlabel("Время, с")
        plt.ylabel("Частота, Гц")
        plt.colorbar(label="дБ")

        plt.tight_layout()
        plt.show()

    # ---------- Вывод ----------
    df_peaks = pd.DataFrame({"Frequency [Hz]": peak_freqs, "Amplitude": peak_amps})
    df_peaks.to_csv("vibration_peaks.csv", index=False)

    print("\n📊 Метрики сигнала:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    print("\n🩺 Диагностика:")
    for k, v in diagnosis.items():
        print(f"  {k}: {'⚠️' if v else '—'}")

    print("\n✅ Сохранён файл vibration_peaks.csv с основными пиками.")
    return {"metrics": metrics, "diagnosis": diagnosis, "peaks": df_peaks}


# ---------- Тестовый пример ----------
if __name__ == "__main__":
    fs = 2000  # частота дискретизации, Гц
    duration = 5  # секунд
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    
    rpm = 1800  # обороты насоса
    f1 = rpm / 60.0
    f2 = 2 * f1

    # Синтетический сигнал с 1x и 2x компонентами + шум
    x = 0.8*np.sin(2*np.pi*f1*t) + 0.4*np.sin(2*np.pi*f2*t) + 0.1*np.random.randn(len(t))

    result = analyze_vibration(x, fs, rpm)
