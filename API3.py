from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import shutil
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import uvicorn

app = FastAPI()

UPLOAD_DIR = Path("Clasificacion")
UPLOAD_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@app.post("/predict/")
async def upload_image(file: UploadFile = File(...)):
    import torch

# Cargar el modelo y mapearlo a la CPU
    model = torch.load('D:/Downloads/INTELIGENCIA ARTIFICIAL/PARCIAL 2/PUNTO 3/modelo_3.pkl', map_location=torch.device('cpu'))
    model = model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    class_names = ['apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bird','boat','book','bottle',
               'bowling_ball','broccoli','bus','cake','car','carrot','cat','cattle','chair','clock','cloud','computer','couch',
               'cow','crab','crocodile','cup','dinosaur','dog','donut','drum','elephant','flatfish','forest','fox','frog','girl',
               'hamster','house','kangaroo','keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man','map',
               'motorcycle','mouse','mushroom','otter','palm_tree','pear','penguin','pickup_truck','pizza','platypus','pomegranate',
               'porcupine','rabbit','raccoon','ray','road','rocket','sandwich','saucer','scorpion','seal','shark','sheep','skateboard',
                'skull','snail','snake','snowman','sofa','spider','squirrel','starfish','strawberry','suitcase','sunglasses','table',
                'tiger','toaster','tortoise','traffic_light','train','truck','umbrella','whale','zebra']

    ruta = UPLOAD_DIR / file.filename
    with ruta.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    imagen = Image.open(ruta)
    imagen = imagen.convert("RGB")
    imagen = transform(imagen)
    imagen = imagen.unsqueeze(0)
    imagen = imagen.to(device)
    
    with torch.no_grad():
        outputs = model(imagen)
        _, pred = torch.max(outputs, 1)

    ruta.unlink()
    return JSONResponse({"Prediccion": class_names[pred.item()]})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)