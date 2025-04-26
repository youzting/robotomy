// TTS 예제 (텍스트 -> 음성)
// 생성된 음성 파일은 예제 코드와 같은 폴더에 위치함
// 각 언어별 모델 사용
// 한국어 모델 지원 안 함 이슈로 한국어는 다국어 모델 사용
// 영어, 한국어 테스트 통과
// langdetect를 통해 text 언어 추정 후 해당 언어로 TTS 생성

from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs
add_safe_globals({XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs})

from TTS.api import TTS
from langdetect import detect

# text 언어 추정 함수
def get_language(text):
    return detect(text)

def get_tts_model(language):
    if language == 'ja':  # 일본어
        model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
    elif language == 'en':  # 영어
        model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
    else: #한국어
        model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)  # 기본 모델
    return model

# TTS 음성 합성 함수
def synthesize_speech(text):
    language = get_language(text)
    tts_model = get_tts_model(language)
    if language == 'ja':
        tts_model.tts_to_file(
            text,
            file_path="output_ja.wav",
            speaker_wav="audio_test_jp.mp3",   # 여기에 음성 샘플 경로
            language="ja"
        )
    elif language == 'en':
        tts_model.tts_to_file(
            text,
            file_path="output_en.wav",
            speaker_wav="audio_test_en.mp3",   # 여기에 음성 샘플 경로
        )
    elif language == 'ko':
        tts_model.tts_to_file(
            text,
            file_path="output_ko.wav",
            speaker_wav="audio_test_ko.wav",   # 여기에 음성 샘플 경로
            language="ko"
        )

text = "도레미파솔라시도레미파솔라시"
synthesize_speech(text)
