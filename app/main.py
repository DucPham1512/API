import logging

from fastapi import FastAPI

from app.interface.router import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

app = FastAPI(title="VSR API", version="0.1.0")
app.include_router(router)
