from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import time, asyncio
from collections import deque
from threading import Lock

CLASSIFICATION_SERVER_URL = "http://localhost:8001/classify"
BATCH_SIZE = 5
MAX_WAIT = 0.1

app = FastAPI(
    title="Classification Proxy",
    description="Proxy server that handles rate limiting and retries for the code classification service"
)

class ProxyRequest(BaseModel):
    """Request model for single text classification"""
    sequence: str

class ProxyResponse(BaseModel):
    """Response model containing classification result ('code' or 'not code')"""
    result: str

queue = asyncio.Queue()
request_queue = deque()
response_dict = {}
proxy_lock = Lock()

@app.post("/proxy_classify")
async def submit_string(req: ProxyRequest):
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    await queue.put((req.sequence, future))
    return await future

@app.on_event("startup")
async def startup():
    asyncio.create_task(batch_worker())
    
async def batch_worker():
    while True:
        batch = []
        futures = []
        start_time = asyncio.get_event_loop().time()

        # Collect up to 5 items, or wait for timeout
        while len(batch) < BATCH_SIZE:
            try:
                timeout = max(0, MAX_WAIT - (asyncio.get_event_loop().time() - start_time))
                item, future = await asyncio.wait_for(queue.get(), timeout)
                batch.append(item)
                futures.append(future)
            except asyncio.TimeoutError:
                break

        if not batch:
            continue

        # Pad batch with empty strings if needed
        while len(batch) < BATCH_SIZE:
            batch.append("")

        # Send batch to backend
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(CLASSIFICATION_SERVER_URL, json={"sequences": batch})
                results = response.json()["results"]
            except Exception as e:
                # Error handling: inform all futures
                for f in futures:
                    f.set_result({"error": str(e)})
                continue

        # Respond to clients individually
        for i, f in enumerate(futures):
            f.set_result({"result": results[i]})