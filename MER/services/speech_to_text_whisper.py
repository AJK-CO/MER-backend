import whisper





def get_text():
# Load model
    model = whisper.load_model("medium")
    audio=whisper.load_audio()
    # Transcribe noisy audio
    result = model.transcribe(audio=audio)
    print(result["text"])
    return result["text"]



if __name__=="__main__":
    get_text()