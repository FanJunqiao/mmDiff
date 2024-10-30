import glob
import os
import numpy as np

def sliding_window(index_type, frames_together, sliding, input_path = None, frame = None, identifier = None, slicing_markers= []):
    r"""
    This function aggregate multiple frames of radar point clouds or radar voxel images by a sliding window. Two type of indexing are supported 
    for aggregating frames: by "file" or by "idx".

    Args:
        index_type (str): Indicating type of frame indexing. Selected from ["file", "idx"]
            
        frame_together (int): Indicating how many frames are aggregated together. 

        sliding (int): Indicating the stride of the sliding window. 

    File-indexing args:
        input_path (str): File path to current frame of data. 

        frame (int): Current frame id.

        identifier (str): file structure identifier. E.g., identifier = "frame\_".
    
    File-indexing return:
        A list of radar data paths.

    Idx-indexing args:
        slicing_marksers (int list): a list of sequence length (total frame number) for each sensor data stream. 

    Idx-indexing return:
        An 2D np.array of shape (n_sample, frame_together), indicating the idx table for selecting dataset data.

    """
    def sliding_window_file(frames_together, sliding, input_path = None, frame = None, identifier = None):
        radar_path = input_path
        path_list = [radar_path]
        for i in range(frames_together-1):
            path = radar_path.replace(str(frame), str(frame - (frames_together-i)*sliding))
            if os.path.exists(path) == False: path = radar_path.replace(identifier+str(frame), identifier+str(frame+i*sliding))
            path_list.append(path)
        return path_list

    def sliding_window_idx(frames_together, sliding, slicing_markers):
        idxs=[]
        start = 0
        for length in slicing_markers:
            end = start + length
            max_len = (length - frames_together) // sliding + 1
            idx_local = np.arange(0, max_len) * sliding + start
            idxs.append(idx_local)
            start = end
        idxs = np.concatenate(idxs)

        return idxs

    if index_type == "file":
        assert input_path != None and frame != None and identifier != None
        return sliding_window_file(frames_together, sliding, input_path, frame, identifier)
    elif index_type == "idx":
        assert len(slicing_markers) != 0
        return sliding_window_idx(frames_together, sliding, slicing_markers)
    else:
        raise NotImplementedError






    