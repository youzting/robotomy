// TTS 예제 (텍스트 -> 음성)
// 생성된 음성 파일은 예제 코드와 같은 폴더에 위치함
// 각 언어별 모델 사용
// 한국어는 다국어 모델 사용
// 영어, 한국어, 일본어, 중국어(간체) 테스트 통과
// langdetect를 통해 text 언어 추정 후 해당 언어로 TTS 생성
// TTS 자동 재생

from torch.serialization import add_safe_globals
from collections import defaultdict  # defaultdict 추가
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs
from TTS.utils.radam import RAdam

# 안전한 글로벌 등록
add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig,
    XttsArgs,
    RAdam,
    defaultdict,
    dict
])

from TTS.api import TTS
from langdetect import detect
import torch
import soundfile as sf
import sounddevice as sd
import os

# 언어 감지 함수
def get_language(text):
    try:
        return detect(text)
    except Exception:
        return 'en'

# 언어별 TTS 모델 반환
def get_tts_model(language):
    if language == 'ja':
        model = TTS(model_name="tts_models/ja/kokoro/tacotron2-DDC", progress_bar=False, gpu=False)
    elif language == 'en':
        model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
    elif language == 'ko':
        model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
    elif language == 'zh-cn':
        model = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST", progress_bar=False, gpu=False)
    else:
        print(f"[!] 언어 '{language}'은 지원되지 않으므로 영어 모델로 대체합니다.")
        model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
        language = 'en'
    return model, language

# 음성 파일 재생 함수
def play_audio(file_name):
    if os.path.exists(file_name):
        data, samplerate = sf.read(file_name)
        sd.play(data, samplerate)
        sd.wait()
    else:
        print(f"[!] 파일을 찾을 수 없습니다: {file_name}")

# TTS 음성 합성 및 저장
def synthesize_speech(text):
    language = get_language(text)
    tts_model, language = get_tts_model(language)

    file_name = f"output_{language}.wav"
    speaker_wavs = {
        'ja': "audio_test_ja.mp3",
        'en': "audio_test_en.mp3",
        'ko': "audio_test_ko.wav",
        'zh-cn': "audio_test_zh-cn.mp3"
    }

    kwargs = {
        "text": text,
        "file_path": file_name,
        "speaker_wav": speaker_wavs.get(language, None)
    }

    if language == 'ko':
        kwargs["language"] = "ko"

    tts_model.tts_to_file(**kwargs)
    print(f"[✔] 음성 파일 저장 완료: {file_name}")

    # 자동 재생
    play_audio(file_name)

# 테스트 실행
if __name__ == "__main__":
    text = "你好。今天天气晴朗。"
    synthesize_speech(text)
