import glob
import os
import numpy as np



def cropping(radar_pc, x_range=[-1.67, 1.67], y_range = [-1, 1], z_range=[-1.67, 1.67]):
    r"""
    This function crop the input radar point cloud according to bounding box defined by (x_range, y_range, z_range). 
    Given input radar point cloud, the points within the bounding box will be selected as the output point cloud and the remaining will be removed.
    
    Input: 
        radar_pc (np.ndarray): a frame of PC with shape [N, 5] or [N, 6] depending on mmWave device, where N is the number of points.
    
    Args:
        radar_pc (np.ndarray): Input radar point cloud. Only support float numpy array.

        x_range (list with length 2): [minimum x range, maximum x range]. Default [-1.67, 1.67]. Modify according to mmWave device.

        y_range (list with length 2): [minimum y range, maximum y range]. Default [-1.67, 1.67]. Modify according to mmWave device.
        
        z_range (list with length 2): [minimum z range, maximum z range]. Default [-1.67, 1.67]. Modify according to mmWave device.

    Return:
        radar_pc [np.ndarray] The returned radar_pc is cropped version of input, of [N', 5] or [N', 6] shape, where N' is the number of points after cropping, not an uniform size.

    Example
        >>> import pysensing.mmwave.PC.preprocessing.uniform as U
        >>> cropped = U.cropping(np.random(500, 5), x_range=[-1.67, 1.67], y_range = [0, 5.3], z_range=[-1.67, 1.67])
    """
    # print(radar_pc)

    radar_pc = radar_pc[np.where((radar_pc[:, 0] <= x_range[1]) & (radar_pc[:, 0] >= x_range[0]))]
    radar_pc = radar_pc[np.where((radar_pc[:,1] <= y_range[1]) & (radar_pc[:,1] >= y_range[0]))]
    radar_pc = radar_pc[np.where((radar_pc[:,2] <= z_range[1]) & (radar_pc[:,2] >= z_range[0]))]

    return radar_pc



def padding(radar_pc, npoint=3000):

    r"""
    This function pad the input radar point cloud into n npoints.
    Given input radar point cloud with N points, if N < n, add (n-N) null points (0,0,0,0,0)
    to the point cloud; else if N > n, select the first n point as the output point cloud.

    Input: 
        radar_pc (np.ndarray): a frame of PC with shape [N, 5] or [N, 6] depending on mmWave device, where N is the number of points.
    
    Args:
        radar_pc (np.ndarray): Input radar point cloud. Only support float numpy array.

        npoint (int, npoint > 0): The number of output point cloud n. Default 3000. 
    
    Return:
        radar_pc (np.ndarray) The returned radar_pc is cropped version of input, of [N', 5] or [N', 6] shape, where N' = npoint.

    Example
        >>> import pysensing.mmwave.PC.preprocessing.uniform as U
        >>> padded = U.padding(np.random(300, 5), npoint=500)
    """
    assert radar_pc.shape[1] == 5 or radar_pc.shape[1] == 6
    Npoint = radar_pc.shape[0]
    if Npoint < npoint:
        for i in range(npoint-Npoint):
            radar_pc = np.append(radar_pc, np.zeros(radar_pc.shape[1]).reshape(1,-1), axis=0)
    else:
        radar_pc = radar_pc[:npoint, :]
    return radar_pc



def voxelizing(radar_pc, x_points, y_points, z_points, x_range = None, y_range = None, z_range = None):

    r"""
    This function converted the a sequence of input radar point clouds into a sequence of voxel images given resolution defined by (x_points, y_points, z_points).
    The value of voxel image represents the number of points within the bin range.
    
    Input: 
        radar_pc: A dict of variable-length radar point clouds, representing a sequence of radar point clouds, e.t. {[N, 5], ..., [N, 5]}, 
        where N is the number of points in each frame, and N is variable length.
    
    Args:
        radar_pc (dict of list): A dict of python list of shape [N, 5] or [N, 6] representing radar pc of each frame.   

        x_points (int): Resolution of voxel image in x-dimension.  

        y_points (int): Resolution of voxel image in y-dimension.    

        z_points (int): Resolution of voxel image in z-dimension.         

        x_range (list with length 2): [minimum x range, maximum x range]. Default None. Using [min(x), max(x)] as the default range.

        y_range (list with length 2): [minimum y range, maximum y range]. Default None. Using [min(y), max(y)] as the default range.
        
        z_range (list with length 2): [minimum z range, maximum z range]. Default None. Using [min(z), max(z)] as the default range.

    Return:
        voxel (np.ndarray): The returned voxel image is the corresponding sequence voxel image, of [x_points, y_points, z_points] shape.

    Example
    >>> import pysensing.mmwave.PC.preprocessing.uniform as U
    >>> voxel = U.voxalizing(np.random(500, 5), x_points = 32, y_points = 32, z_points = 10)
    """


    #y and z points in this cluster of frames
    x = radar_pc[:,0]
    y = radar_pc[:,1]
    z = radar_pc[:,2]

    if x_range == None:
        x_min = np.min(x)
        x_max = np.max(x)
    else:
        x_min = x_range[0]
        x_max = x_range[1]

    if y_range == None:
        y_min = np.min(y)
        y_max = np.max(y)
    else:
        y_min = y_range[0]
        y_max = y_range[1]

    if z_range == None:
        z_min = np.min(z)
        z_max = np.max(z)
    else:
        z_min = z_range[0]
        z_max = z_range[1]
    z_res = (z_max - z_min)/z_points
    y_res = (y_max - y_min)/y_points
    x_res = (x_max - x_min)/x_points

    pixel = np.zeros([x_points,y_points,z_points])

    x_current = x_min
    y_current = y_min
    z_current = z_min

    x_prev = x_min
    y_prev = y_min
    z_prev = z_min


    x_count = 0
    y_count = 0
    z_count = 0


    for i in range(y.shape[0]):
        x_current = x_min
        x_prev = x_min
        x_count = 0
        done=False

        while x_current <= x_max and x_count < x_points and done==False:
            y_prev = y_min
            y_current = y_min
            y_count = 0
            while y_current <= y_max and y_count < y_points and done==False:
                z_prev = z_min
                z_current = z_min
                z_count = 0
                while z_current <= z_max and z_count < z_points and done==False:
                    if x[i] < x_current and y[i] < y_current and z[i] < z_current and x[i] >= x_prev and y[i] >= y_prev and z[i] >= z_prev:
                        pixel[x_count,y_count,z_count] = pixel[x_count,y_count,z_count] + 1
                        done = True

                        #velocity_voxel[x_count,y_count,z_count] = velocity_voxel[x_count,y_count,z_count] + velocity[i]
                    z_prev = z_current
                    z_current = z_current + z_res
                    z_count = z_count + 1
                y_prev = y_current
                y_current = y_current + y_res
                y_count = y_count + 1
            x_prev = x_current
            x_current = x_current + x_res
            x_count = x_count + 1
    return pixel


def load_from_raw(parent_dir, sub_dirs, file_ext='*.txt'):
    r"""
    This function load .txt format raw data collected from TI mmwave radar groups into a sequence of voxel images given resolution 
    defined by (x_points, y_points, z_points). The value of voxel image represents the number of points within the bin range.
    
    Args:
        parent_dir (str): Path to dataset train split or test split.

        sub_dirs (str): Path to data file recorded the sensor data stream.

        file_ext (str): Type of file. Default "\*.txt", supporting TI mmWave radar groups.

    Return:
        voxels_all (np.ndarray): Returned sequence of voxel images.

        slicing_marker (np.ndarray): slicing_marker indicates the length of sensor data in one single sequence, i.e. how many frames contained by one file.

        labels_all (np.ndarray): Returned label of the action corresponding to each voxel images.
    
    """
    print(sub_dirs)
    voxels_all = np.empty((0, 10, 32, 32) )
    slicing_marker = []
    labels_all = []
    
    

    for sub_dir in sub_dirs:
        files=sorted(glob.glob(os.path.join(parent_dir,sub_dir, file_ext)))
        for fn in files:
            print(fn)
            print(sub_dir)
            action_data = get_data(fn) # get a dict of variable length PC data stored by list
            print(len(action_data))

            '''
            Voxelization of all data obtained
            '''
            
            voxels = []
            # Now for 2 second windows, we need to club together the frames and we will have some sliding windows
            for i in action_data:
                pc = action_data[i]
                pc = np.array(pc)

                pix = voxelizing(pc, 10, 32, 32)
                #print(i, f.shape,pix.shape)
                voxels.append(pix)
                

            voxels = np.array(voxels)
            for i in range(voxels.shape[0]):
                    labels_all.append(sub_dir)
            voxels_all = np.vstack([voxels_all,voxels])
            
            slicing_marker.append(voxels.shape[0])
            print(voxels_all.shape, len(slicing_marker), slicing_marker[-1], len(labels_all), labels_all[-1])
            Data_path =  os.path.join(parent_dir, "data.npz")
            np.savez(Data_path, voxels_all, np.array(slicing_marker), np.array(labels_all))





    return voxels_all, slicing_marker, labels_all


def get_data(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    frame_num_count = -1
    frame_num = []
    x = []
    y = []
    z = []
    velocity = []
    intensity = []
    wordlist = []


    for x1 in lines:
        for word in x1.split():
            wordlist.append(word)

    length1 = len(wordlist)

    for i in range(0,length1):
        if wordlist[i] == "point_id:" and wordlist[i+1] == "0":
            frame_num_count += 1
        if wordlist[i] == "point_id:":
            frame_num.append(frame_num_count)
        if wordlist[i] == "x:":
            x.append(wordlist[i+1])
        if wordlist[i] == "y:":
            y.append(wordlist[i+1])
        if wordlist[i] == "z:":
            z.append(wordlist[i+1])
        if wordlist[i] == "velocity:":
            velocity.append(wordlist[i+1])
        if wordlist[i] == "intensity:":
            intensity.append(wordlist[i+1])

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    frame_num = np.asarray(frame_num)
    velocity = np.asarray(velocity)
    intensity = np.asarray(intensity)

    x = x.astype(np.float)
    y = y.astype(np.float)
    z = z.astype(np.float)
    velocity = velocity.astype(np.float)
    intensity = intensity.astype(np.float)
    frame_num = frame_num.astype(np.int)


    data = dict()

    for i in range(len(frame_num)):
        if int(frame_num[i]) in data:
            data[frame_num[i]].append([x[i],y[i],z[i],velocity[i],intensity[i]])
        else:
            data[frame_num[i]]=[]
            data[frame_num[i]].append([x[i],y[i],z[i],velocity[i],intensity[i]])

    return data





