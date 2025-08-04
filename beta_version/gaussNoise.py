from PIL import Image
import numpy as np

# original image
og = np.array(Image.open('penguin.jpg'), dtype=np.float32) / 255.0

# create gaussian noise
mean = 0
var = 0.10
sigma = np.sqrt(var)
noise = np.random.normal(loc=mean, 
                     scale=sigma, 
                     size=og.shape)

# add a gaussian noise
corrupted = np.clip(og + noise, 0.0, 1.0)
corrupted_pil = Image.fromarray((corrupted * 255).astype(np.uint8))
corrupted_pil.show()
