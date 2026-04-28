from fastapi import FastAPI

from app.interface.router import router

app = FastAPI(title="VSR API", version="0.1.0")
app.include_router(router)
