from uvicorn import run

if __name__ == '__main__':
    run("api:app", reload=True)