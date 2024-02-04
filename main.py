from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# if __name__ == '__main__':
#     uvicorn.run("main:app", host='0.0.0.0', post=8080, reload=True, workers=3)