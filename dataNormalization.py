import numpy as np
import librosa
import scipy.signal
import soundfile as sf


def normalize_amplitude(audio):
    return audio / np.max(np.abs(audio))


def resample_audio(audio, original_sr, target_sr):
    return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)


def remove_noise(audio, sr):
    b, a = scipy.signal.butter(N=10, Wn=1000 / (sr / 2), btype="highpass")
    return scipy.signal.filtfilt(b, a, audio)


def trim_or_pad_audio(audio, length):
    if len(audio) > length:
        return audio[:length]
    elif len(audio) < length:
        return np.pad(audio, (0, length - len(audio)), "constant")
    return audio


def trim_silence(audio, sr):
    return librosa.effects.trim(audio, top_db=30, frame_length=2048, hop_length=512)[0]


def bandpass_filter(audio, sr, low, high):
    nyquist = sr // 2
    low = low / nyquist
    high = high / nyquist
    b, a = scipy.signal.butter(N=4, Wn=[low, high], btype="band")
    return scipy.signal.lfilter(b, a, audio)


import os
import librosa

for i in range(1, 3):
    # 파일 경로 설정
    if i == 1:
        directory = "./raw_data/male"
        processed_directory = "./normalized_data/male"
    else:
        directory = "./raw_data/female"
        processed_directory = "./normalized_data/female"
    # 파일 목록 가져오기
    file_list = [f for f in os.listdir(directory) if f.endswith(".wav")]
    # 처리된 데이터를 저장할 폴더 생성
    os.makedirs(processed_directory, exist_ok=True)

    for file in file_list:
        # 파일 불러오기
        path = os.path.join(directory, file)
        audio, sr = librosa.load(path, sr=None)

        # 정규화 및 처리 과정 수행
        audio = normalize_amplitude(audio)
        audio = resample_audio(audio, sr, target_sr=16000)  # 16000 Hz로 리샘플링
        # audio = remove_noise(audio, 16000)  # 노이즈 제거보다 데이터 손실이 더 크기에 일단 제외
        audio = trim_silence(audio, 16000)
        audio = bandpass_filter(audio, 16000, low=300, high=3400)  # 300~3400 Hz 대역 필터링
        audio = trim_or_pad_audio(audio, length=20000)  # 2초 길이로 정규화

        # 처리된 데이터 저장
        processed_path = os.path.join(processed_directory, file)
        sf.write(processed_path, audio, 16000)
