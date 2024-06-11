import numpy as np
import glob
from discretization import adaptive_grid_theta, adaptive_grid_xy, adaptive_control_grid_v, adaptive_control_grid_w

# Example grid definitions
error_threshold = 0.5
fine_spacing = 0.1
coarse_spacing = 0.6

nt = 100
ex_space = adaptive_grid_xy()    
ey_space = adaptive_grid_xy()
eth_space = adaptive_grid_theta()
v_space = adaptive_control_grid_v()
w_space = adaptive_control_grid_w()

# Initialize transition matrix with smaller chunk
chunk_size = 10  # Adjust this value as needed

# Define the number of chunks and the shape of each chunk
num_chunks = nt // chunk_size
chunk_shape = (chunk_size, len(ex_space), len(ey_space), len(eth_space), 
               len(v_space), len(w_space), 6, 4)

# Initialize the final combined transition matrix
combined_transition_matrix = np.zeros((nt, len(ex_space), len(ey_space), len(eth_space), 
                                       len(v_space), len(w_space), 6, 4), dtype=np.float16)

# Load each chunk and combine them
for chunk_idx in range(num_chunks):
    chunk_filename = f'transition_matrix_float16_chunk_{chunk_idx}.npz'
    with np.load(chunk_filename) as data:
        chunk_data = data['arr_0']
        combined_transition_matrix[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size, ...] = chunk_data

# Now combined_transition_matrix contains the entire transition matrix
# You can save the combined matrix if needed
np.savez_compressed('combined_transition_matrix_float16.npz', combined_transition_matrix)
