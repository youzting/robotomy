// Whisper 예제 (음성 -> 텍스트)
// https://ttsmp3.com/ 해당 사이트에서 입력한 문장을 mp3 파일로 다운로드 후, 예제 코드와 같은 폴더에 위치시킴.

import whisper

model = whisper.load_model("base")  # base, small, medium, large 중 선택 가능

result = model.transcribe("audio.mp3")  # 파일 경로 수정

print("📝 Transcription:")
print(result["text"])
