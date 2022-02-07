import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
import io
from PIL import Image
import os
from scipy.stats import stats
import matplotlib.image as mpimg
from matplotlib.image import imread

img = cv2.cvtColor(cv2.imread('Data/stereo_37_L.jpg'), cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

img.shape

#splitting into channels
blue, green, red = cv2.split(img)
# Plotting the images
fig = plt.figure(figsize=(15, 7.2))
fig.add_subplot(131)
plt.title("Blue Channel")
plt.imshow(blue)
fig.add_subplot(132)
plt.title("Green Channel")
plt.imshow(green)
fig.add_subplot(133)
plt.title("Red Channel")
plt.imshow(red)
plt.show()

blue_temp_df = pd.DataFrame(data=blue)
blue_temp_df

df_blue = blue/255
df_green = green/255
df_red = red/255

#visualize all principal components
pca_b_orig = PCA(n_components=720)
pca_b_orig.fit(df_blue)
pca_g_orig = PCA(n_components=720)
pca_g_orig.fit(df_green)
pca_r_orig = PCA(n_components=720)
pca_r_orig.fit(df_red)

#plot explained variance
plt.grid()
plt.plot(np.cumsum(pca_b_orig.explained_variance_ratio_ * 100))
plt.xlabel('Number of components')
plt.ylabel('Explained variance Blue')
plt.grid()
plt.plot(np.cumsum(pca_g_orig.explained_variance_ratio_ * 100))
plt.xlabel('Number of components')
plt.ylabel('Explained variance Green')
plt.grid()
plt.plot(np.cumsum(pca_r_orig.explained_variance_ratio_ * 100))
plt.xlabel('Number of components')
plt.ylabel('Explained variance Red')

# reduce the components
pca_b = PCA(n_components=200)
pca_b.fit(df_blue)
trans_pca_b = pca_b.transform(df_blue)
pca_g = PCA(n_components=200)
pca_g.fit(df_green)
trans_pca_g = pca_g.transform(df_green)
pca_r = PCA(n_components=200)
pca_r.fit(df_red)
trans_pca_r = pca_r.transform(df_red)

#printing out the variance accuracy/results
print(trans_pca_b.shape) #get the bits size of image from this
print(trans_pca_r.shape)
print(trans_pca_g.shape)

print(f"Blue Channel : {sum(pca_b.explained_variance_ratio_)}")
print(f"Green Channel: {sum(pca_g.explained_variance_ratio_)}")
print(f"Red Channel  : {sum(pca_r.explained_variance_ratio_)}")

fig = plt.figure(figsize=(15, 7.2))
fig.add_subplot(131)
plt.title("Blue Channel")
plt.ylabel('Variation explained')
plt.xlabel('Eigen Value')
plt.bar(list(range(1, 201)), pca_b.explained_variance_ratio_)
fig.add_subplot(132)
plt.title("Green Channel")
plt.ylabel('Variation explained')
plt.xlabel('Eigen Value')
plt.bar(list(range(1, 201)), pca_g.explained_variance_ratio_)
fig.add_subplot(133)
plt.title("Red Channel")
plt.ylabel('Variation explained')
plt.xlabel('Eigen Value')
plt.bar(list(range(1, 201)), pca_r.explained_variance_ratio_)
plt.show()

#reconstruct image and visualize by inversing
b_arr = pca_b.inverse_transform(trans_pca_b)
g_arr = pca_g.inverse_transform(trans_pca_g)
r_arr = pca_r.inverse_transform(trans_pca_r)
print(b_arr.shape, g_arr.shape, r_arr.shape)

#merge all the channels into 1
img_reduced = (cv2.merge((b_arr, g_arr, r_arr)))
img_reduced_priorInverse = (cv2.merge((trans_pca_b, trans_pca_g, trans_pca_r))) #merging image before inverse
print(img_reduced.shape)

fig = plt.figure(figsize=(10, 7.2))
fig.add_subplot(121)
plt.title("Original Image")
plt.imshow(img)
fig.add_subplot(122)
plt.title("Reduced Image")
plt.imshow(img_reduced)
plt.show()

cv2.imwrite('stereo_37_L_r.jpg', img_reduced * 255)
cv2.imshow('reduced', img_reduced)
im = Image.open('Data/stereo_37_L.jpg')
im_resize = im.resize((500, 500))
buf = io.BytesIO()
im_resize.save(buf, format='JPEG')
byte_im = buf.getvalue()

img_reduced_priorInverse = (img_reduced_priorInverse * 255)
cv2.imwrite("Data/stereo_37_L_compressed.jpg", img_reduced_priorInverse) #the compressed image that has all three channels
cv2.waitKey(0)
# closing all open windows
cv2.destroyAllWindows()

#using reduced saved image, inverse it to decompress the original image
img_reduced_priorInverse = imread('Data/stereo_37_L_compressed.jpg')
blue_c,green_c,red_c = cv2.split(img_reduced_priorInverse)
b_arr_c = pca_b.inverse_transform(blue_c)
g_arr_c = pca_g.inverse_transform(green_c)
r_arr_c = pca_r.inverse_transform(red_c)

img_reconstruct = (cv2.merge((b_arr_c, g_arr_c, r_arr_c)))
fig = plt.figure(figsize=(10, 7.2))
fig.add_subplot(121)
plt.title("Original Image")
plt.imshow(img)
fig.add_subplot(122)
plt.title("Reconstructed Image")
plt.imshow(img_reconstruct)
plt.show()

cv2.imwrite("Data/stereo_37_L_reconstructed.jpg", img_reconstruct) #the reconstructed image that has all three channels

print(img.nbytes)
print(img_reconstruct.nbytes) # x8 the original amount because it's stored in floats (0.0-1.0) not ints (0-255)
print(img_reduced_priorInverse.nbytes) # 6.4x decrease (unsure about size on disk)

# closing all open windows
cv2.destroyAllWindows()
