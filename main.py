from fastapi import FastAPI , Response
import uvicorn
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from fastapi.responses import FileResponse
import io
from pydantic import BaseModel

token = ""
app = FastAPI()

with open('token.txt') as f:
    token = f.readlines()[0]

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=token)
pipe = pipe.to(device)

class Req(BaseModel):
    prompt: str
    seed: Union[int, None] = None
    height: Union[int, None] = None
    width: Union[int, None] = None
    guidance_scale: Union[float, None] = None



@app.get("/getimage")
async def getimage(prompt = "astronaut riding a horse"):
     with autocast("cuda"):
         image = pipe(prompt, guidance_scale=7.5)["sample"][0]
    
     bytes_image = io.BytesIO()
     image.save(bytes_image, format='PNG')

     return Response(content = bytes_image.getvalue(), media_type="image/png")

@app.post("/getimage")
async def getimage(request : Req):
     with autocast("cuda"):
         image = pipe(request.prompt, guidance_scale=7.5)["sample"][0]
    
     bytes_image = io.BytesIO()
     image.save(bytes_image, format='PNG')

     return Response(content = bytes_image.getvalue(), media_type="image/png")

@app.get("/")
async def heartbeat():
    return {"message" : "OK"}

uvicorn.run(app, host="0.0.0.0", port=8000)