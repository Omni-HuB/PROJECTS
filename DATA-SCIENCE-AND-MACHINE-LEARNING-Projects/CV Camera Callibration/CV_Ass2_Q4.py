# %%


# %%
# """
# Performing camera calibration by 
# passing the value of known 3D points (objpoints)
# and corresponding pixel coordinates of the 
# detected corners (imgpoints)
# """
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# print("Camera matrix : \n")
# print(mtx)
# print("dist : \n")
# print(dist)
# print("rvecs : \n")
# print(rvecs)
# print("tvecs : \n")
# print(tvecs)

# %%


# %%


# %%


# %%
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


# %%
# Defining the dimensions of checkerboard
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 


# %%
# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

print(len(objpoints))

# Extracting path of individual image stored in a given directory
images = glob.glob('./Q4_ChessBoard_Dataset/*.jpg')
sorted(images)

total_images = len(images)

print(f"Total Images:{total_images}")

# %%
print(images)

# %%


for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.title(f'Image {idx+1}')
    plt.axis('off')
    plt.show()
    
    # img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
h,w = img.shape[:2]
    
    


# %%


# %%
"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, camera_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print("Camera matrix : \n")
print(camera_mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

# %%
# Reporting intrinsic parameters
focal_length_x = camera_mtx[0, 0]
focal_length_y = camera_mtx[1, 1]
principal_point_x = camera_mtx[0, 2]
principal_point_y = camera_mtx[1, 2]
skew = camera_mtx[0, 1]  # This value is expected to be zero or very close to zero

print("Intrinsic Camera Parameters:")
print(f"Focal Lengths: fx = {focal_length_x}, fy = {focal_length_y}")
print(f"Principal Point: cx = {principal_point_x}, cy = {principal_point_y}")
print(f"Skew: {skew} (usually close to zero and not reported unless significant)")
print("")

# Reporting extrinsic parameters for each image
for i in range(len(rvecs)):
    rotation_vector = rvecs[i]
    translation_vector = tvecs[i]
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)  # Convert rotation vector to rotation matrix
    
    print(f"Extrinsic Parameters for Image {i+1}:")
    print("Rotation Matrix:")
    print(rotation_matrix)
    print("Translation Vector:")
    print(translation_vector)
    print("")



# %%
# #Q4 part 3

# # Using the derived camera parameters to undistort the image

# img = cv2.imread(images[0])
# # Refining the camera matrix using parameters obtained by calibration
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_mtx, dist, (w,h), 1, (w,h))

# # Method 1 to undistort the image
# dst = cv2.undistort(img, camera_mtx, dist, None, newcameramtx)

# # Method 2 to undistort the image
# mapx,mapy=cv2.initUndistortRectifyMap(camera_mtx,dist,None,newcameramtx,(w,h),5)

# dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# # Displaying the undistorted image
# cv2.imshow("undistorted image",dst)
# cv2.waitKey(0)



import matplotlib.pyplot as plt



# Let's iterate through the first five images to undistort them and display
for i, fname in enumerate(images[:5]):
    img = cv2.imread(fname)  # Read the image
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_mtx, dist, (w,h), 1, (w,h))

    # Undistort using the new camera matrix
    dst = cv2.undistort(img, camera_mtx, dist, None, new_camera_mtx)
    
    # Convert to RGB for displaying with matplotlib
    dst_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    
    # Plot the undistorted images
    plt.figure(figsize=(10, 6))
    plt.imshow(dst_rgb)
    plt.title(f'Undistorted Image {i+1}')
    plt.axis('off')
    plt.show()

# # Report the estimated radial distortion coefficients
# radial_distortion_coefficients = dist[:3]  # First three coefficients
# print(f"Radial Distortion Coefficients: {radial_distortion_coefficients}")

radial_distortion_coefficients = dist[0][:3]
print(f"Radial Distortion Coefficients: k1 = {radial_distortion_coefficients[0]}, k2 = {radial_distortion_coefficients[1]}, k3 = {radial_distortion_coefficients[2]}")





# %%
#Q4 part4

mean_error = 0
errors = []
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
    errors.append(error)
mean_error /= len(objpoints)

print(len(objpoints))
# print(objpoints)


mean_reprojection_error = np.mean(errors)
stddev_reprojection_error = np.std(errors)

plt.bar(range(len(errors)), errors)
plt.xlabel('Image index')
plt.ylabel('Reprojection error')
plt.title('Reprojection error for each of the 25 images')
plt.show()

print(f" Mean of the re-projection error : {mean_reprojection_error}")
print(f" Standard Deviation of the re-projection error: {stddev_reprojection_error}")

# %%
# Ensure that you only have data for the 25 images
assert len(objpoints) == 25
assert len(imgpoints) == 25

# Now calculate the reprojection errors
mean_error = 0
errors = []
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
    errors.append(error)
mean_error /= len(objpoints)

# Calculate the mean and standard deviation of the reprojection errors
mean_reprojection_error = np.mean(errors)
stddev_reprojection_error = np.std(errors)

# Plot the errors
plt.figure(figsize=(10, 6)) # Adjust the size as needed
plt.bar(range(1, len(errors)+1), errors)  # Start the index at 1 for better readability
plt.xticks(range(1, len(errors)+1))  # Adjust the x-ticks to match image indices
plt.xlabel('Image index')
plt.ylabel('Reprojection error')
plt.title('Reprojection error for each of the 25 images')
plt.show()

# Print the mean and standard deviation
print(f"Mean of the reprojection error: {mean_reprojection_error}")
print(f"Standard Deviation of the reprojection error: {stddev_reprojection_error}")


# %%
for i in range(len(objpoints)):
    # Read the image and convert it to RGB for matplotlib compatibility
    img = cv2.imread(images[i])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw the detected corners
    cv2.drawChessboardCorners(img, (CHECKERBOARD[0], CHECKERBOARD[1]), imgpoints[i], True)
    
    # Project the 3D object points to 2D image points
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_mtx, dist)
    
    # Draw the reprojected corners
    imgpoints2 = np.rint(imgpoints2).astype(int)  # Convert to integer for drawing
    for p in imgpoints2:
        cv2.circle(img_rgb, (p[0][0], p[0][1]), 5, (255, 0, 0), -1)  # Blue circles
    
    # Plot the image with both the original and reprojected points
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.title(f'Image {i+1} with Detected and Reprojected Points')
    plt.axis('off')
    plt.show()


print(len(objpoints))

# %%
normals = []

for i in range(len(rvecs)):
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvecs[i])
    
    # The z-axis in the object's coordinate frame
    z_axis = np.array([[0], [0], [1]])
    
    # The normal to the checkerboard in the camera coordinate frame
    normal = np.dot(R, z_axis)
    
    
    normals.append(normal)
    print(f"Normal for image {i+1}: {normal.ravel()}")


# %%



