import cv2

# Read the query image as query_img
# and train image This query image
# is what you need to find in train image
# Save it in the same directory
# with the name image.jpg 
query_img = cv2.imread('calculator_1.jpg')
train_img = cv2.imread('calculator_4.jpg')
  
# Convert it to grayscale
query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
  
# Initialize the ORB detector algorithm
orb = cv2.ORB_create(nfeatures=5000)
  
# Now detect the keypoints and compute
# the descriptors for the query image
# and train image
#queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
#trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)

queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)
