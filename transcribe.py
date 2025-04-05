// Whisper 예제 (음성 -> 텍스트)
// 음성 파일은 예제 코드와 같은 폴더에 위치시킴.

import whisper

model = whisper.load_model("base")  # base, small, medium, large 중 선택 가능

result = model.transcribe("audio.mp3")  # 파일 이름 수정

print("📝 Transcription:")
print(result["text"])
