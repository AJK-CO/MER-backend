

from fastapi import APIRouter,File, UploadFile

from fastapi.responses import FileResponse

from fastapi.responses import HTMLResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from MER.services.test_model import find_emotion
import shutil
import os
from MER.services.speech_to_text_whisper import get_text

router=APIRouter(tags=["SER"],prefix="/ser")


# This API is used for recording th audio of the user and save it 
@router.post("/save-audio")
def recorde_audioasync(file: UploadFile = File(...)):
      
    filename = file.filename
    # Ensure the file has a .wav extension
    if not filename.endswith('.wav'):
        filename += '.wav'
    
    file_location = os.path.join(r"MER\services", filename)
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    
    return find_emotion(r"MER\services\recording.wav")



@router.get("/get-text")
def html_page():
    return get_text("MER/services/recording.wav") 

@router.get("/")
def html_page():
    return FileResponse(r"templates\index.html")



