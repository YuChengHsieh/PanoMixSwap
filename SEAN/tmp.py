from PIL import Image
import glob
import numpy as np
import cv2
label_path = glob.glob('../myResearch/out/target_segmap_rmFurtinure.png')
label = cv2.imread(label_path[0])

label = cv2.resize(label,(int(label.shape[1]/2),int(label.shape[0]/2)),interpolation=cv2.INTER_NEAREST)
import pdb; pdb.set_trace()
# label = Image.open(label_path[0])
# label = label.resize((int(label.size[0]/2),int(label.size[1]/2)),Image.NEAREST )
# label_data = np.array(label.getdata())
# import pdb; pdb.set_trace()



