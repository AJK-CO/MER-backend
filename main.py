
from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
from MER.services import fer_model
from MER.routers.ser import router as ser


from MER.routers.fer import router as fer
from MER.routers.ter import router as ter

from MER.routers.mer import router as mer
import uvicorn

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(ser)

app.include_router(fer)

app.include_router(ter)
app.include_router(mer)

if __name__=="__main__":
    uvicorn.run("main:app",reload=True)
