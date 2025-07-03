import numpy as np
import os
from skimage import io
from imageio import imread#, imwrite as imsave
import re
from MachineLearningLibrary import resizeImageAndLandmarks, registerAndCutImage, registerPointClouds2D, poly2mask

baseDataPath = './demo_data/olderSCD/'
landmarkPath = './demo_data/olderSCD/'
newDatabasePath = './demo_results/StandardizedImages_olderSCD'
referenceImagePath = './demo_data/olderSCD/07739.jpg'
referenceLandmarksPath = './demo_data/olderSCD/07739.pts'

# If the background will be removed
removeBackground = True

# Size of the images
cnnImageSize = (256, 256, 3)

############################################
## Reading the reference image and landmarks
############################################
referenceImage = io.imread(referenceImagePath)
referenceLandmarks = np.ndarray(shape=[1,44,2])
with open(referenceLandmarksPath) as landmarksFile:
    lines = landmarksFile.readlines()
    for p in range(len(lines)):
        s = re.split(',| |\t|\n', lines[p])
        referenceLandmarks[0][p][0] = float(s[0])
        referenceLandmarks[0][p][1] = float(s[1])

referenceImage, referenceLandmarks = resizeImageAndLandmarks(referenceImage, referenceLandmarks, cnnImageSize, totalPadding=0)

# Creating folder structure for the output database path if it doesn't exist
if not os.path.exists(newDatabasePath):
    os.makedirs(newDatabasePath)

index =0
for file in os.listdir(baseDataPath):
    if file.endswith('jpg'):
        im = imread(os.path.join(baseDataPath, file))

        coords = np.ndarray(shape=[1,44,2])
        with open(landmarkPath + file.replace('.jpg', '.pts')) as landmarksFile:
            lines = landmarksFile.readlines()
            for p in range(44):
                s = re.split(',| |\t|\n', lines[p])
                coords[0][p][0] = float(s[0])
                coords[0][p][1] = float(s[1])

        index += 1
        print('{:05d}'.format(index), end='\r')

        image, coords_reg = registerAndCutImage(im, coords, referenceLandmarks, cnnImageSize)

        # Setting regions outside the face to zero
        if removeBackground:

            polygonCoords = np.concatenate([coords_reg[0,33:,0:1], coords_reg[0,33:,1:2]], axis=1)

            # Extending
            polygonCoords[0, :] += 50 * (polygonCoords[0, :] - polygonCoords[1, :]) / np.linalg.norm(polygonCoords[0, :] - polygonCoords[1, :])
            polygonCoords[10, :] += 50 * (polygonCoords[10, :] - polygonCoords[9, :]) / np.linalg.norm(polygonCoords[10, :] - polygonCoords[9, :])

            mask = poly2mask(polygonCoords[:,1], polygonCoords[:,0], image.shape[:2])
            mask = np.tile(np.expand_dims(mask.astype(np.uint8), axis=2), (1, 1, 3))
            image *= mask

        io.imsave(os.path.join(newDatabasePath, file), image)
