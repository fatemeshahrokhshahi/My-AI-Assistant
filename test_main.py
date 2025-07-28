# test_main.py - Simple test version to check if basic structure works

from fastapi import FastAPI

app = FastAPI(title="Test App")

@app.get("/")
def root():
    return {"message": "Basic FastAPI structure test - Working!"}

@app.get("/test")
def test():
    return {"status": "Structure is working correctly"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)