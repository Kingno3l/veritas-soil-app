from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Soil Health Monitoring API is working!"}
