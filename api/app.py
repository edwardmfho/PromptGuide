from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import Dict, Tuple

from prompt_generation import refine_prompt
from dotenv import load_dotenv
import uvicorn


load_dotenv()
origins = ["*"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

request_counts: Dict[str, Tuple[int, datetime]] = {}

async def rate_limiter(request: Request):
    ip = request.client.host
    now = datetime.now()
    count, timestamp = request_counts.get(ip, (0, now))
    
    # Reset count if more than 24 hours have passed
    if now - timestamp > timedelta(days=1):
        count = 0
    
    # Increment count
    count += 1
    request_counts[ip] = (count, now)
    
    if count > 10:
        raise HTTPException(status_code=429, detail="Request limit exceeded")
    
    return True  # Return value isn't used, but you could return something if needed


@app.get("/")
def read_root():
    return {"Hello": "World"}
# https://ohmeow.com/guides/fastapi#error-nginx-reverse-proxy-https
@app.post("/v1/prompt/generate/", dependencies=[Depends(rate_limiter)])
def generate_prompt(prompt:str):
    results = refine_prompt(prompt)
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)