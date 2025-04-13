// TTS 예제 (텍스트 -> 음성)
// 생성된 음성 파일은 예제 코드와 같은 폴더에 위치함
// 다국어 지원 모델 사용
// 영어, 한국어 테스트 통과

from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs
add_safe_globals({XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs})

from TTS.api import TTS

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)

tts.tts_to_file(
    text="안녕하세요, 만나서 반갑습니다. 좋은 하루 되세요.",
    file_path="output.wav",
    speaker_wav="audio_test_ko.wav",   # 여기에 음성 샘플 경로
    language="ko"
)

print("✅ 음성 생성 완료: output.wav")
