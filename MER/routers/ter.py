

from fastapi import APIRouter,Request

from fastapi.responses import FileResponse

from fastapi.responses import FileResponse

from MER.services.text_model_service import predict_text_emotion

from MER.schema.text_input import TextRequest
router=APIRouter(tags=["TER"],prefix="/ter")



# This API is used for recording th audio of the user and save it 
@router.post("/predict-emotion")
def get_text(input: TextRequest):
    print(input.text)
 
    return predict_text_emotion(input.text)

@router.get("")
def html_page():
    return FileResponse(r"templates\ter.html")




