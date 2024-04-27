import cv2
import glob
from PIL import Image
import numpy as np
#from torch.nn import functional as F
#import torch
#semantic_labels = glob.glob('./structured3D_inference/semantic_label/*')
GAN_results = glob.glob('results_pipeline/structure3D/test_latest/images/synthesized_image/*')
#original_images = glob.glob('./structured3D_inference/original_image/*')
"""
semantic_labels = ['/media/public_dataset/structured3D/structure_panoDR/Structured3D/test/scene_03300/2D_rendering/190739/panorama/full/semantic.png',
                       '/media/public_dataset/structured3D/structure_panoDR/Structured3D/test/scene_03366/2D_rendering/907634/panorama/full/semantic.png',
                       '/media/public_dataset/structured3D/structure_panoDR/Structured3D/test/scene_03303/2D_rendering/38/panorama/full/semantic.png']
original_images = ['/media/public_dataset/structured3D/structure_panoDR/Structured3D/test/scene_03300/2D_rendering/190739/panorama/full/rgb_warmlight.png',
                       '/media/public_dataset/structured3D/structure_panoDR/Structured3D/test/scene_03366/2D_rendering/907634/panorama/full/rgb_coldlight.png',
                       '/media/public_dataset/structured3D/structure_panoDR/Structured3D/test/scene_03303/2D_rendering/38/panorama/full/rgb_warmlight.png']
"""

semantic_labels = ['./structured3D_inference/styleImage_semanticLabel/1.png']
original_images = ['./structured3D_inference/style_image/1.png']

#import pdb; pdb.set_trace()

for i in range(len(semantic_labels)):
    sem = Image.open(semantic_labels[i])
    sem = sem.resize((int(sem.size[0]/2),int(sem.size[1]/2)))
    sem = np.asarray(sem)
    gan = Image.open(GAN_results[i])
    gan = gan.convert('RGB')
    #
    gan_copy = np.asarray(gan).copy()
    orig = Image.open(original_images[i])
    orig = orig.convert('RGB')
    
    orig = orig.resize((int(orig.size[0]/2),int(orig.size[1]/2)))
    orig = np.asarray(orig)
    
    #import pdb; pdb.set_trace()
    for j in range(sem.shape[0]):
        for k in range(sem.shape[1]):
            if sem[j][k] != 22 and sem[j][k] != 1 and sem[j][k] != 2:
                gan_copy[j][k] = orig[j][k]
                #import pdb ; pdb.set_trace()
    a = i+1
    #import pdb; pdb.set_trace()
    gan_copy = cv2.cvtColor(gan_copy,cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'results_pipeline/structure3D/test_latest/images/final_processed_image/{a}.png',gan_copy)
    #save_img = Image.new(gan.mode,gan.size)
    #import pdb; pdb.set_trace()
    #save_img.putdata(gan_copy)
    #save_img.save(f'results/structure3D/test_latest/images/final_processed_image/{a}.png')
    #import pdb; pdb.set_trace()