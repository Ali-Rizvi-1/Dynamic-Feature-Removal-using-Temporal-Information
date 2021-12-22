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

cv2.imshow("Image", train_img_gray)
cv2.waitKey(0)