from fastapi import APIRouter

from fastapi import Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from MER.services.vedio_services import gen_frames

router=APIRouter(
    prefix="/fer",
    tags=['Facial Emotion recogition']
)

templates = Jinja2Templates(directory="templates")


@router.get('')
def index(request: Request):
    return templates.TemplateResponse(name="show_video.html",context= {"request": request})


@router.get('/video_feed')
def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')
