import os
import os.path
import MachineLearningLibrary
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.exposure

## Configuration
workingFolder = './demo_results'

figSize=(10,10)
feature = 'Texture at center of cupid’s bow (R1)'
resolution = 0 # 0 for R1, 1 for R2, 2 for R3
landmarkID = 26
"""Landmark IDs including 7 in the midline of the face and 13 symmetric landmarks
0: lateral canthi
1: lower eyelids
2: medial canthi
3: upper eyelids
4: center of the pupil
10: lateral of nose root
11: alar crease
12: center of ala
13: bottom of ala
14: nostril top
15: columella
21: nasion
22: tip of nose
23: philtrum
24: oral commissures
25: side of cupid's bow
26: center of cupid's bow
30: side of lower lip
31: lower border of upper lip
32: upper border of lower lip
"""

textureImagesFolder = os.path.join(workingFolder, 'TextureImages')
if not os.path.exists(textureImagesFolder):
    os.makedirs(textureImagesFolder)

mGeneObject = MachineLearningLibrary.ML()
mGeneObject.negativeClassName = 'youngerSCD' # Class label 0 (e.g., control)
mGeneObject.positiveClassName = 'olderSCD' # Class label 1 (e.g., sickle cell or syndromic)

imageList = mGeneObject.GetTextureAtLandmark(workingFolder, landmarkID)
negativeImage = imageList[resolution]
positiveImage = imageList[3 + resolution]

numberOfRows = 1 # First row is the context. The second row is the LBPs in the smaller image

center = (positiveImage.shape[0]-1)/2
nElements = mGeneObject.LBP_list[resolution,0]
radius = mGeneObject.LBP_list[resolution,1]
Ws = mGeneObject.Ws
smallCenter = radius+(Ws-1)/2

f=plt.figure(num=1, figsize=figSize)

ax=plt.subplot(numberOfRows, 2, 1)
plt.imshow(skimage.exposure.equalize_hist(negativeImage, nbins=5), cmap='gray')
plt.plot(center,center,'.r')
ax.add_patch(patches.Ellipse((center, center), 2*radius+Ws, 2*radius+Ws,fill=False, edgecolor='green', linestyle='solid'))
plt.axis('off')
plt.title('Context: ' + mGeneObject.negativeClassName)

ax=plt.subplot(numberOfRows, 2, 2)
plt.imshow(skimage.exposure.equalize_hist(positiveImage, nbins=5), cmap='gray')
# plt.imshow(positiveImage, cmap='gray')
plt.plot(center, center,'.r')
ax.add_patch(patches.Ellipse((center, center), 2*radius+Ws, 2*radius+Ws,fill=False, edgecolor='green', linestyle='solid'))
plt.axis('off')
plt.title('Context: ' + mGeneObject.positiveClassName)

print('Saving: ' + os.path.join(textureImagesFolder, feature+'.png'))
plt.savefig(os.path.join(textureImagesFolder, feature+'.png'), dpi=600)
