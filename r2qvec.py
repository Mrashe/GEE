import numpy as np
# R = np.array([[ 0.63568407, -0.03804422,  0.77101129],
#  [ 0.15893528,  0.98383658, -0.08249338],
#  [-0.75541071,  0.17498063,  0.63145581]])
#R is the rotation matrix

R = np.array([[-0.80411332,  0.03676329,  0.5933382 ],
 [-0.06602357, -0.99743416, -0.02767641],
 [-0.59079832 , 0.06142927, -0.80447734]] )

# [[-0.80411332  0.03676329  0.5933382 ]
#  [-0.06602357 -0.99743416 -0.02767641]
#  [-0.59079832  0.06142927 -0.80447734]] 


# rotation_matrix = np.load("rotation_matrix.npy")
rotation_matrix = R

det = np.linalg.det(rotation_matrix)
if(det<0 and np.isclose(det,-1.0,atol=1e-06)):
    rotation_matrix = -rotation_matrix
print(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2])
qw = 0.5 * np.sqrt(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2])
qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * qw)
qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * qw)
qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * qw)
qvec = np.array([qw,qx,qy,qz])
print(qvec)