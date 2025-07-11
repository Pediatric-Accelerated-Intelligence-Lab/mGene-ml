import os, numpy, PIL
from PIL import Image

path = './demo_results/StandardizedImages_olderSCD/'
path2 = './demo_results/'

# Access all PNG files in directory
allfiles=os.listdir(path)
imlist=[filename for filename in allfiles if  filename[-4:] in [".jpg",".JPG"]]

# Assuming all images are the same size, get dimensions of first image
w,h=Image.open(path + imlist[0]).size
N=len(imlist)

# Create a numpy array of floats to store the average (assume RGB images)
arr=numpy.zeros((h,w,3),numpy.float)
# arr=numpy.zeros((h,w),numpy.float)

# Build up average pixel intensities, casting each image as an array of floats
for im in imlist:
    imarr=numpy.array(Image.open(path + im),dtype=numpy.float)
    arr=arr+imarr/N

# Round values in array and cast as 8-bit integer
arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)

# Generate, save and preview final image
# out=Image.fromarray(arr,mode="RGB")
out=Image.fromarray(arr)
out.save(path2 + "Avg_face_olderSCD.jpg")
# out.show()
