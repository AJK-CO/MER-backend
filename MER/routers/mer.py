

from fastapi import APIRouter, HTTPException,Request, File, UploadFile

from fastapi.responses import FileResponse

from fastapi.responses import FileResponse
import os
from concurrent.futures import ThreadPoolExecutor
from MER.services.text_model_service import predict_text_emotion
from fastapi.responses import JSONResponse
from pydub import AudioSegment
import subprocess
from MER.schema.text_input import TextRequest
from MER.services.video_services import get_fer_emotion
from MER.services.test_model import find_emotion
from MER.services.speech_to_text_whisper import get_text

router=APIRouter(tags=["MER"],prefix="/mer")


async def save_file(file: UploadFile, file_path: str):
    try:
        with open(file_path, "wb") as media_file:
            media_file.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save the uploaded file: {str(e)}")

def run_ffmpeg_command(command):
    try:
        # Capture the output of the FFmpeg command
        result = subprocess.run(command, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {str(e)}")
    
@router.post("/save-media")
async def save_media(file: UploadFile = File(...)):
    # File paths
    media_filename = f"audios/{file.filename}"
    video_filename = f"videos/{os.path.splitext(file.filename)[0]}.mp4"
    audio_filename_wav = f"audios/{os.path.splitext(file.filename)[0]}.wav"

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(media_filename), exist_ok=True)
    os.makedirs(os.path.dirname(video_filename), exist_ok=True)
    os.makedirs(os.path.dirname(audio_filename_wav), exist_ok=True)

    # Save the uploaded file asynchronously
    await save_file(file, media_filename)

    # FFmpeg commands for video and audio extraction
    video_command = [
        'ffmpeg', '-y', '-i', media_filename, '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '22', video_filename
    ]
    audio_command = [
        'ffmpeg', '-y', '-i', media_filename, '-vn', '-acodec', 'pcm_s16le',
        '-ar', '48000', '-ac', '1',  # '-ac', '1' ensures mono audio
        audio_filename_wav
    ]

    # Run FFmpeg commands in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        video_future = executor.submit(run_ffmpeg_command, video_command)
        audio_future = executor.submit(run_ffmpeg_command, audio_command)

        # Wait for both tasks to complete
        try:
            video_future.result()
            audio_future.result()
        except HTTPException as e:
            return JSONResponse(content={"error": e.detail}, status_code=500)

    # Check the duration of the saved audio file to prevent processing errors
    audio_duration_command = [
        'ffmpeg', '-i', audio_filename_wav, '-f', 'null', '-'
    ]
    audio_duration_result = run_ffmpeg_command(audio_duration_command)

    if "Duration: 00:00:00" in audio_duration_result:
        return JSONResponse(content={"error": "Audio file is too short for processing."}, status_code=400)
    
    return "Audio File saved successfully"

@router.get("/fer")
def get_fer_output():
    fer=get_fer_emotion()
    return fer

@router.get("/ser_ter")
def get_fer_output():
    ser= find_emotion("Audios/recording.wav")
    text=get_text("Audios/recording.wav") 
    ter=predict_text_emotion(text)
    return {"ser":ser,
            "ter":ter,
            "text":text}

@router.get("")
def html_page():
    return FileResponse("templates/mer.html")




