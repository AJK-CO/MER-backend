import whisper

def get_text(path):
# Load model
    model = whisper.load_model("medium")
    audio=whisper.load_audio(path)
    # Transcribe noisy audio
    result = model.transcribe(audio=audio)
    print(result["text"])
    return result["text"]

if __name__=="__main__":
    get_text()