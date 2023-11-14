import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft, ifft


def compute_cepstrum(signal):
    # 신호에 FFT를 적용해 스펙트럼을 계산
    spectrum = fft(signal)
    # 스펙트럼의 로그 절댓값을 계산
    log_amplitude = np.log(np.absolute(spectrum))
    # 로그 스펙트럼에 IFFT를 적용해 켑스트럼을 계산
    cepstrum = np.abs(ifft(log_amplitude))
    return cepstrum


def compute_human_cepstrum(xs):
    cepstrum = compute_cepstrum(xs)
    quefrencies = np.array(range(len(xs))) / 16000

    # Filter values that are not within human pitch range
    # highest frequency
    period_lb = 1 / 270
    # lowest frequency
    period_ub = 1 / 70

    cepstrum_filtered = []
    quefrencies_filtered = []
    for i, quefrency in enumerate(quefrencies):
        if quefrency < period_lb or quefrency > period_ub:
            continue

        quefrencies_filtered.append(quefrency)
        cepstrum_filtered.append(cepstrum[i])

    return cepstrum_filtered


def plot_signal_and_cepstrum(signal, cepstrum, fs, title):
    time = np.arange(len(signal)) / fs
    quefrency = np.arange(len(cepstrum)) / fs

    plt.figure(figsize=(12, 6))

    # Plot the time-domain signal
    plt.subplot(2, 1, 1)
    plt.plot(time, signal)
    plt.title(f"Time-Domain Signal - {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Plot the cepstrum
    plt.subplot(2, 1, 2)
    plt.plot(quefrency, cepstrum)
    plt.title(f"Cepstrum - {title}")
    plt.xlabel("Quefrency (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


fs1, a = wavfile.read("./normalized_data/male/male_0_a.wav")
fs2, e = wavfile.read("./normalized_data/male/male_0_e.wav")
fs3, i = wavfile.read("./normalized_data/male/male_0_i.wav")
fs4, o = wavfile.read("./normalized_data/male/male_0_o.wav")
fs5, u = wavfile.read("./normalized_data/male/male_0_u.wav")
cepstrum1 = compute_human_cepstrum(compute_cepstrum(a))
cepstrum2 = compute_human_cepstrum(compute_cepstrum(e))
cepstrum3 = compute_human_cepstrum(compute_cepstrum(i))
cepstrum4 = compute_human_cepstrum(compute_cepstrum(o))
cepstrum5 = compute_human_cepstrum(compute_cepstrum(u))
plot_signal_and_cepstrum(a, cepstrum1, fs1, "a")
plot_signal_and_cepstrum(e, cepstrum2, fs2, "e")
plot_signal_and_cepstrum(i, cepstrum3, fs3, "i")
plot_signal_and_cepstrum(o, cepstrum4, fs4, "o")
plot_signal_and_cepstrum(u, cepstrum5, fs5, "u")

fs1, a = wavfile.read("./normalized_data/female/female_4_a.wav")
fs2, e = wavfile.read("./normalized_data/female/female_4_e.wav")
fs3, i = wavfile.read("./normalized_data/female/female_4_i.wav")
fs4, o = wavfile.read("./normalized_data/female/female_4_o.wav")
fs5, u = wavfile.read("./normalized_data/female/female_4_u.wav")

cepstrum1 = compute_human_cepstrum(compute_cepstrum(a))
cepstrum2 = compute_human_cepstrum(compute_cepstrum(e))
cepstrum3 = compute_human_cepstrum(compute_cepstrum(i))
cepstrum4 = compute_human_cepstrum(compute_cepstrum(o))
cepstrum5 = compute_human_cepstrum(compute_cepstrum(u))

plot_signal_and_cepstrum(a, cepstrum1, fs1, "a")
plot_signal_and_cepstrum(e, cepstrum2, fs2, "e")
plot_signal_and_cepstrum(i, cepstrum3, fs3, "i")
plot_signal_and_cepstrum(o, cepstrum4, fs4, "o")
plot_signal_and_cepstrum(u, cepstrum5, fs5, "u")
