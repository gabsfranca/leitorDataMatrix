import cv2 
import numpy as np
import os 
from PIL import Image

def deixaBarulhenta(image):
    linha, col, ch = image.shape
    mean = 0 
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (linha, col, ch))
    gauss = gauss.reshape(linha, col, ch)
    barulhenta = image + gauss
    barulhenta = np.clip(barulhenta, 0, 255).astype(np.uint8)
    return barulhenta

def rodaImagem(image, angulo, scale = 1.0):
    (h, w) = image.shape[:2]
    centro = (w/2, h/2)
    M = cv2.getRotationMatrix2D(centro, angulo, scale)
    rodada = cv2.warpAffine(image, M, (int(w * scale), int(h * scale)))
    return rodada

def gif2png(camGif):
    img = Image.open(camGif)
    camPng = camGif.replace('.gif', '.png')
    img.save(camPng)
    return camPng

def zoaVariosGifs(arrayComGifs):
    for camGif in arrayComGifs:
        nomeCerto = os.path.basename(camGif).replace('.gif', '')
        dir = f"zoada_{nomeCerto}"
        os.makedirs(dir, exist_ok=True)
        camPng = gif2png(camGif)
        img = cv2.imread(camPng)
        if img is None:
            print(f"erro ao carregar img: {camPng}")
            continue
        for i in range(20):
            imgBarulhenta = deixaBarulhenta(img)
            imgRodada = rodaImagem(imgBarulhenta, np.random.uniform(-40, 40))
            cv2.imwrite(os.path.join(dir, f'{nomeCerto}_{i}.png'), imgRodada)

camimgs = ["src\\dataset\\teste\\111\\111_18.png", "src\\dataset\\teste\\111\\111_19.png", "src\dataset\teste\111\111.png"]

zoaVariosGifs(camimgs)