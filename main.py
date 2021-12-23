import cv2

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
  
# draw the matches to the final image
# containing both the images the drawMatches()
# function takes both images and keypoints
# and outputs the matched query image with
# its train image
final_img = cv2.drawMatches(query_img, queryKeypoints,
train_img, trainKeypoints, matches[:100],None)
  
final_img = cv2.resize(final_img, (1000,650))

# saving an image
cv2.imwrite('FeatureMatchesImage.jpg',final_img)

# # Show the final image
# cv2.imshow("Image", final_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()