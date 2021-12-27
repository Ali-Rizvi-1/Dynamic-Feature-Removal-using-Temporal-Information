# all the imports 
import cv2
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns; sns.set()  # for plot styling
from sklearn.cluster import KMeans
import numpy as np
import math

# Read the calculator frame1 as query_img and frame 4 as train image 
# The features in query image is what you need to find in train image

query_img = cv2.imread('calculator_frame1.jpg')
train_img = cv2.imread('calculator_frame4.jpg')
  
# Convert it to grayscale
query_img_gray = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
train_img_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
  
# Initialize the ORB detector algorithm
orb = cv2.ORB_create(nfeatures=2000)
  
# Now detect the keypoints and compute the descriptors for the query image
# and train image

queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_gray,None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_gray,None)

# showing an image using cv2 imshow method
# cv2.imshow("Image", train_img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# checking the length of train and query image keypoints
print(len(queryKeypoints), len(trainKeypoints))

# Initialize the Matcher for matching the keypoints and then match the keypoints
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(queryDescriptors,trainDescriptors)
  
# draw the matches to the final image containing both the images the drawMatches()
# function takes both images and keypoints and outputs the matched query image with its train image
final_img = cv2.drawMatches(query_img, queryKeypoints,
train_img, trainKeypoints, matches[:100],None)
  
final_img = cv2.resize(final_img, (1000,650))

# saving an image
# cv2.imwrite('FeatureMatchesImage.jpg',final_img)

print("Total matched features in the two frames are",len(matches))

# computing the feature translations in the form of vectors
MatchedQueryPoints = []
MatchedTrainPoints = []
translation_vectors = []
for i in matches:
  queryIndex = i.queryIdx
  trainIndex = i.trainIdx 
  MatchedQueryPoints.append(queryKeypoints[queryIndex])
  MatchedTrainPoints.append(trainKeypoints[trainIndex])
  translation_vectors.append((trainKeypoints[trainIndex].pt[0] - queryKeypoints[queryIndex].pt[0], trainKeypoints[trainIndex].pt[1] - queryKeypoints[queryIndex].pt[1]))

X_data = []
Y_data = []
for vec in translation_vectors:
    X_data.append(vec[0])
    Y_data.append(vec[1])

# plotting the x data and y data as a scatter plot
plt.scatter(X_data,Y_data)
plt.show()

# converting the x and y data into polar coordinates
R_data = []
theta_data = []
list_polarCoordinates = []

for i in range(len(X_data)):
  R_data.append((X_data[i]**2+Y_data[i]**2)**0.5)
  if X_data[i] == 0:
    theta_data.append((math.pi)/2) 
  else:
    theta_data.append(math.atan(Y_data[i]/X_data[i]))
  list_polarCoordinates.append([theta_data[-1],R_data[-1]])

plt.scatter(theta_data,R_data)
plt.show()

data_array = np.array(list_polarCoordinates)

# k-mean clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(data_array)
y_kmeans = kmeans.predict(data_array)

# plotting polar coordinates after clustering
fig = plt.figure()
plt.scatter(data_array[:, 0], data_array[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()
# save plot to file
fig.savefig('kmean_clustering.png')

# finding the center with lowest radius value to identify the static feature points in the frame
find = []
for mean_point in centers:
  find.append(mean_point[1])
min_center = min(find)
min_index = find.index(min_center)

selected_keypoints = []
selected_descriptors = []

for i in range(len(y_kmeans)):
  if y_kmeans[i] == min_index:
    trainIndex = matches[i].trainIdx
    selected_keypoints.append(trainKeypoints[trainIndex])
    #selected_descriptors.append(trainDescriptors[trainIndex])

# plotting the matched feature points on the current image frame
trainImg_Matchedfeatures = cv2.drawKeypoints(train_img, MatchedTrainPoints, np.array([]), (0,0,255), 4)
cv2.imwrite('ImagewithDynamicFeatures.jpg',trainImg_Matchedfeatures)
cv2.imshow("Image", trainImg_Matchedfeatures)
cv2.waitKey(3000)
cv2.destroyAllWindows()

# plotting the selected feature points on the current image frame
trainImg_Selectedfeatures = cv2.drawKeypoints(train_img, selected_keypoints, np.array([]), (0,0,255), 4)
cv2.imwrite('ImagewithRemovedDynamic.jpg',trainImg_Selectedfeatures)
cv2.imshow("Image", trainImg_Selectedfeatures)
cv2.waitKey(3000)
cv2.destroyAllWindows()