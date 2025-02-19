# %%
%pip install numpy open3d

# %%
import numpy as np
import open3d as o3d
import os


# %%


# %%
# Path to the directory containing the .pcd files

# pcd_dir = 'path_to_your_pcd_files_directory'

pcd_dir = "./lidar_scans"

# Assuming the .pcd files are named in a sequential order
pcd_files = [f for f in os.listdir(pcd_dir) if f.endswith('.pcd')]

# Sort the files to maintain the sequence
pcd_files.sort()


# %%
# Let's list the contents of the 'camera_parameters' directory to see what files are there
camera_parameters_path = "./camera_parameters"
camera_parameters_files = os.listdir("./camera_parameters")

# %%
# Function to read normals from a text file
def read_normals(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        normals = np.array([list(map(float, line.strip().split())) for line in lines])
    return normals
# Initialize a dictionary to hold frame normals
frame_normals = {}

# Iterate over each folder in 'camera_parameters', read the normals, and store them in the dictionary
for folder_name in camera_parameters_files:
    if folder_name.endswith('.jpeg'):  # Ensure we're working with the correct directories
        normals_file_path = os.path.join(camera_parameters_path, folder_name, 'camera_normals.txt')
        if os.path.isfile(normals_file_path):  # Check if the normals file exists
            normals = read_normals(normals_file_path)
            frame_normals[folder_name] = normals

# For demonstration, let's print the normals for one of the frames
sample_frame = list(frame_normals.keys())[0]
frame_normals[sample_frame]



# %%
# Function to read normals from a text file
def read_rotation_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        normals = np.array([list(map(float, line.strip().split())) for line in lines])
    return normals
# Initialize a dictionary to hold frame normals
frames_rotation_matrix = {}

# Iterate over each folder in 'camera_parameters', read the rotation_matrix, and store them in the dictionary
for folder_name in camera_parameters_files:
    if folder_name.endswith('.jpeg'):  # Ensure we're working with the correct directories
        
        cam_normals_file_path = os.path.join(camera_parameters_path, folder_name, 'camera_normals.txt')
        cam_rotation_matrix_file_path = os.path.join(camera_parameters_path, folder_name, 'rotation_matrix.txt')
        cam_rotation_vector_file_path = os.path.join(camera_parameters_path, folder_name, 'rotation_vectors.txt')
        cam_translation_vector_file_path = os.path.join(camera_parameters_path, folder_name, 'translation_vectors.txt')
        
        if os.path.isfile(cam_rotation_matrix_file_path):  # Check if the normals file exists
            rotation_matrix = read_rotation_matrix(cam_rotation_matrix_file_path)
            frames_rotation_matrix[folder_name] = rotation_matrix

# For demonstration, let's print the normals for one of the frames
sample_frame = list(frames_rotation_matrix.keys())[0]
frames_rotation_matrix[sample_frame]


# %%
camera_parameters_files

# %%
pcd_files

# for i in range(len(pcd_files)) :
#     print(pcd_files[i])

# %%
frame_normals

# %%

def compute_plane_normal_and_offset(points):
    # Subtract the mean to center the points
    points_centered = points - np.mean(points, axis=0)

    # Calculate the covariance matrix
    H = np.dot(points_centered.T, points_centered)

    # Perform SVD
    U, S, Vt = np.linalg.svd(H)

    # The normal to the chessboard plane is the last column of V
    normal = Vt[-1, :]

    # The offset can be found by projecting the mean of the points onto the normal vector
    offset = np.dot(normal, np.mean(points, axis=0))
    
    return normal, offset




# %%

# # Storage for normals and offsets
# normals = []
# offsets = []

# # Process each .pcd file
# for idx, pcd_file in enumerate(pcd_files):
#     try:
#         # Load the point cloud data
#         pcd_path = os.path.join(pcd_dir, pcd_file)
#         pcd = o3d.io.read_point_cloud(pcd_path)

#         # Compute the normal and offset of the plane
#         normal, offset = compute_plane_normal_and_offset(np.asarray(pcd.points))
        
#         # Store the results
#         normals.append(normal)
#         offsets.append(offset)
        
#         print(f'{idx+1} Processed file {pcd_file}: Normal - {normal}')
#         print(f'{idx+1} Processed file {pcd_file}: Offset - {offset}')
        
#     except Exception as e:
#         print(f'An error occurred while processing file {pcd_file}: {e}')

# # At this point, 'normals' and 'offsets' contain the plane parameters for all .pcd files


# %%

# Storage for normals and offsets in a dictionary
LIDAR_plane_parameters = {}

# Process each .pcd file
for idx, pcd_file in enumerate(pcd_files):
    try:
        # Load the point cloud data
        pcd_path = os.path.join(pcd_dir, pcd_file)
        pcd = o3d.io.read_point_cloud(pcd_path)

        # Compute the normal and offset of the plane
        normal, offset = compute_plane_normal_and_offset(np.asarray(pcd.points))
        
        # Store the results in the dictionary with the file name as the key
        LIDAR_plane_parameters[pcd_file] = {'normal': normal, 'offset': offset}
        
        print(f'{idx+1} Processed file {pcd_file}: Normal - {normal}')
        print(f'{idx+1} Processed file {pcd_file}: Offset - {offset}')
        
    except Exception as e:
        print(f'An error occurred while processing file {pcd_file}: {e}')
        
        
print(LIDAR_plane_parameters)

# At this point, 'plane_parameters' contains the plane parameters for all .pcd files keyed by their file names


# %%


# %%


# %% [markdown]
# 

# %%



