import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np

def plot_3d_graph(tensor1, tensor2, edges=None, elev=-45, azim=-135, roll=45, save_path=None):
    if tensor1 is not None:
        if torch.is_tensor(tensor1):
            tensor1 = tensor1.numpy()
        tensor1 = tensor1 - tensor1[:1, :]
        tensor2 = tensor2 - tensor2[:1, :]

    
    # Move tensors to CPU if on GPU
    tensor1_cpu = tensor1
    tensor2_cpu = tensor2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extracting x, z, and inverted y coordinates from the tensors (switching y and z)
    if tensor1 is not None:
        x1, z1, y1 = tensor1_cpu[:, 0], tensor1_cpu[:, 2], -tensor1_cpu[:, 1]
    if tensor2 is not None:
        x2, z2, y2 = tensor2_cpu[:, 0], tensor2_cpu[:, 2], -tensor2_cpu[:, 1]
        if tensor2_cpu.shape[-1] >3:
            colors = np.where(tensor2_cpu[:,4] >= 0, 'green', 'red')
        else:
            colors = "red"

    # Scatter plot for the first tensor
    if tensor1 is not None:
        ax.scatter(x1, z1, y1, c='b', marker='o', label='Tensor 1')

    # Scatter plot for the second tensor
    if tensor2 is not None:
        ax.scatter(x2, z2, y2, c=colors, marker='o', label='Tensor 2')

    # Set default edges if not provided
    if tensor1 is not None:
        if edges is None:
            edges = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                                [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]], dtype=torch.long)

        # Plotting lines based on the edges
        for edge in edges:
            start_node = tensor1_cpu[edge[0]]
            end_node = tensor1_cpu[edge[1]]

            ax.plot([start_node[0], end_node[0]],
                    [start_node[2], end_node[2]],
                    [-start_node[1], -end_node[1]], c='b', linestyle='-', linewidth=2)
            

        if tensor2 is not None and tensor2_cpu.shape[-1] == 2:
            # Plotting lines based on the edges
            for edge in edges:
                start_node = tensor2_cpu[edge[0]]
                end_node = tensor2_cpu[edge[1]]

                ax.plot([start_node[0], end_node[0]],
                        [start_node[2], end_node[2]],
                        [-start_node[1], -end_node[1]], c='r', linestyle='-', linewidth=2)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Z-axis')
    ax.set_zlabel('Y-axis')


    # # Set axis limits to the range [-1, 1]
    if tensor1 is not None:
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

    # Invert the y-axis
    ax.invert_yaxis()

    # # Hide all ticks and labels except for the x, y, and z axes
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    # ax.grid(False)
    # ax.axis('off')

    # ax.set_title('3D Graph Plot')

    # Adjusting the angle of view
    ax.view_init(elev=elev, azim=azim, roll=roll)

    # Save the plot to a file if save_path is provided
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

# # Example usage with saving to a file
# # Replace these tensors with your actual data
# # Assuming you're using PyTorch and the tensors are on GPU
# tensor1 = torch.rand(17, 3, device='cuda') * 2 - 1  # Scale to the range [-1, 1]
# tensor2 = torch.rand(17, 3, device='cuda') * 2 - 1  # Scale to the range [-1, 1]

# # Adjust the elevation and azimuth angles as needed
# elevation_angle = 30
# azimuthal_angle = 45

# # Provide a file path to save the plot (e.g., 'output_graph.png')
# output_file_path = 'output_graph.png'

# # You can provide your custom edges or leave them as None to use the default edges
# # edges = torch.tensor([[...]], dtype=torch.long)
# edges = None

# plot_3d_graph(tensor1, tensor2, edges=edges, elev=elevation_angle, azim=azimuthal_angle, save_path=output_file_path)
