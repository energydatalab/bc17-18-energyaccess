import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

VIIRS_IMAGE_PATH = "PATH HERE"
BINARY_MASK_PATH = "PATH HERE"
VALID_PATH = "PATH HERE"

def load_image_data():
	valid_images_pd = pd.read_csv(VALID_PATH)
	ids = valid_images_pd.ID
	binaryData = np.empty(nvalues, SIZE, 1)
	viirsData = np.empty(nvalues, SIZE, 48)
	nvalues = ids.shape[0]
	for i in range(nvalues);
		cid = ids[i]
		b_f_name = BINARY_MASK_PATH+cid.astype(str)+'.tif'
		v_f_name = VIIRS_IMAGE_PATH+cid.astype(str)+'.tif'
		binaryData[i]= mpl.image.imread(b_f_name)
		viirsData[i] = mpl.image.imread(v_f_name)

	return binaryData, viirsData

def apply_mask(binaryData, viirsData):
	mask_data = np.empty(binaryData.shape[0], binaryData[0].shape[0], binaryData[0].shape[1])
	for i in range(binaryData.shape[0]):
		for j in range(binaryData[0].shape[0]):
			for k in range(binaryData[0].shape[1]):
				if binaryData[i][j][k]==0:
					
			
	

	 
