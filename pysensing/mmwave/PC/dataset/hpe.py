import os 
import numpy as np
import pandas as pd
import yaml
import glob
import torch
import cv2
from torch.utils.data import Dataset

from pysensing.mmwave.PC.preprocessing.uniform import cropping, padding
from pysensing.mmwave.PC.preprocessing.sliding_window import sliding_window


class MMBody(Dataset):
    r"""
    Implementation of "mmBody Benchmark: 3D Body Reconstruction Dataset and Analysis for Millimeter Wave Radar".

    Point cloud Mesh/Pose reconstruction dataset collected 4D imaging radar, see https://arberobotics.com/wp-content/uploads/2021/05/4D-Imaging-radar-product-overview.pdf. 
    2 train scenes ["Lab1", "Lab2"] and 7 test scenes ["Lab1", "Lab2", "Furnished", "Occlusion", "Rain", "Smoke", "Poor_lighting"] are included. Depth and RGB sensors are 
    implemented without calibration (to be implemented). 

    Args:
        root (str of Path): Path to the dataset.

        split (str): Split of the dataset. Selected from ["train", "test"] 

        modalities (list of str): The selected output modalities.

        test_scenario (str): Only applicable for test split dataset, the selected scene (from 7 test scenes) for testing.

        normalized (bool): Whether normalized data using the ground truth human pose (torso location). If True, 
        subtracting all points (radar point clouds, ground truth pose) with the torso location.

    Reference: 
        https://github.com/Chen3110/mmBody
    """
    def __init__(self,
                 root="../../../data/mmpose/",
                 split='train',
                 modalities = ["Depth", "Radar", "RGB"],
                 test_scenario = "Lab1",
                 normalized=True):
        self.root = root
        self.modalities = modalities
        self.normalized = normalized
        if test_scenario == "all": 
            self.scenario = ["lab1", "lab2", "furnished", "rain", "smoke", "poor_lighting", "occlusion"]
        else: 
            self.scenario = [test_scenario]
        self.normalized_center = [None, None, None]
        assert (split == 'train' or split == 'test')

        # initialize dataframe
        self.df = pd.DataFrame()
        self.path_df = pd.DataFrame()


        if split == "train": # Trainloader
            split_path = os.path.join(self.root, split)
            for sub_path in glob.glob(os.path.join(split_path, "*")):
                # If is depth false data, continue
                sequence = int(sub_path.split("/")[-1].split("_")[-1])
                data_path_df = self._load_mesh(sub_path)
                for modality in self.modalities:
                    if modality == "Radar":
                        if "Radar" in data_path_df:
                            continue
                        radar_path_df = self._load_radar(sub_path)
                        data_path_df = data_path_df.set_index(['Sequence', 'Frame']).join(
                            radar_path_df.set_index(['Sequence', 'Frame'])).reset_index(names=['Sequence', 'Frame'])
                    elif modality == "Depth":
                        if "Depth" in data_path_df:
                            continue
                        depth_path_df = self._load_depth(sub_path)
                        data_path_df = data_path_df.set_index(['Sequence', 'Frame']).join(
                            depth_path_df.set_index(['Sequence', 'Frame'])).reset_index(names=['Sequence', 'Frame'])
                    elif modality == "RGB":
                        rgb_path_df = self._load_rgb(sub_path)
                        data_path_df = data_path_df.set_index(['Sequence', 'Frame']).join(
                            rgb_path_df.set_index(['Sequence', 'Frame'])).reset_index(names=['Sequence', 'Frame'])
                    else:
                        raise(RuntimeError)


                self.path_df = pd.concat([self.path_df, data_path_df], ignore_index=True)

        
        else: # Testloader
            for s in self.scenario:
                split_path = os.path.join(self.root, split, s)
                for sub_path in glob.glob(os.path.join(split_path, "*")):
                    data_path_df = self._load_mesh(sub_path)
                    for modality in self.modalities:
                        if modality == "Radar":
                            if "Radar" in data_path_df:
                                continue
                            radar_path_df = self._load_radar(sub_path)
                            data_path_df = data_path_df.set_index(['Sequence', 'Frame']).join(
                                radar_path_df.set_index(['Sequence', 'Frame'])).reset_index(
                                names=['Sequence', 'Frame'])
                        elif modality == "Depth":
                            if "Depth" in data_path_df:
                                continue
                            depth_path_df = self._load_depth(sub_path)
                            data_path_df = data_path_df.set_index(['Sequence', 'Frame']).join(
                                depth_path_df.set_index(['Sequence', 'Frame'])).reset_index(
                                names=['Sequence', 'Frame'])
                        elif modality == "RGB":
                            rgb_path_df = self._load_rgb(sub_path)
                            data_path_df = data_path_df.set_index(['Sequence', 'Frame']).join(
                                rgb_path_df.set_index(['Sequence', 'Frame'])).reset_index(names=['Sequence', 'Frame'])
                        else:
                            raise(RuntimeError)
                    self.path_df = pd.concat([self.path_df, data_path_df], ignore_index=True)






    def __len__(self):
        return self.path_df.shape[0]

    def __getitem__(self, index):
        r'''
        Returns:
            (input, label)[np.ndarray, int]: 
            "input" (np.ndarray): The point clouds from preprocessed radar point clouds. The shape of pc: [frame_together, npoints, 6], default [4, 5000, 6].
            "label" (np.ndarray): The ground truth human pose. The shape of pose: [17, 3].
                    
        Example:
            >>> hpe_train_dataset = MMBody(dataset_root, split='Train', modalities = "Radar")
            >>> index = 9
            >>> sample= har_train_dataset.__getitem__(index)
        '''

        #  Initialize data of format:
        # ["Sequence", "Frame", "Pose", "Pose_hand", "Shape", "Vertices", "Joints", "modality 1", "modality 2"]
        sequence = self.path_df.iloc[index]["Sequence"]
        frame = self.path_df.iloc[index]["Frame"]
        mesh_path = self.path_df.iloc[index]["Mesh"]
        data = {"Sequence": sequence, "Frame": frame}
        

        # Load mesh ground truth
        mesh_data = np.load(mesh_path)
        for k in mesh_data.keys():
            data[f"{k}"] = mesh_data[k]

        
        

        if self.normalized:
            self.normalized_center = data["joints"][0]
 
        for modality in self.modalities:
            if modality == "Radar":
                radar_path = self.path_df.iloc[index]["Radar"]
                radar_path_list = sliding_window(index_type="file", frames_together=4, sliding=1, input_path = radar_path, frame=frame, identifier="frame_")
                radar_data_list = []
                # current implentation
                for path in radar_path_list:
                    radar_data = np.load(path)
                    if self.normalized:
                        radar_data[:, :3] = np.subtract(radar_data[:, :3], self.normalized_center)
                        radar_data[:, 3] = radar_data[:, 3] * 1e38
                        radar_data[:, 4] = radar_data[:, 4] * 10
                        radar_data[:, 5] = radar_data[:, 5] / 100
                    radar_data = cropping(radar_data)
                    radar_data = padding(radar_data, npoint=72*72)
                    
                    radar_data_list.append(radar_data)
                data["Radar"] = np.stack(radar_data_list, axis=0)




            if modality == "Depth":
                depth_path = self.path_df.iloc[index]["Depth"]
                depth_data = cv2.imread(depth_path)
                data["Depth"] = depth_data


        

            if modality == "RGB":

                rgb_path = self.path_df.iloc[index]["RGB"]
                rgb_data = cv2.imread(rgb_path)
                rgb_data = rgb_data.view(3, rgb_data.shape[1], rgb_data.shape[2])
      
                data["RGB"] = rgb_data

            
        # convert to (data, label) pair
        selected_joints =  [0,1, 4, 7, 2, 5, 8, 6, 12, 15, 24,  16, 18,  20, 17, 19, 21]
        label =  torch.from_numpy(data["joints"][selected_joints,:])
        input = torch.from_numpy(data["Radar"])

        return (input, label)




    def _load_rgb(self, current_path):
        """
        Description: Load the all rgb data from .png to np.array(), concatenate all data
        into dataframe. Sort the dataframe according to the frame.

        Input:
        self: inherent from class
        current_path: indicated the current path of the data to be loaded.
        (Example): ../data/mmpose/train/sequence_9

        Output:
        Return a Tuple of format (sequence, frame, rgb data)
        """

        rgb_path = []
        sequence = int(current_path.split("/")[-1].split("_")[-1])
        for file in glob.glob(os.path.join(current_path, "image", "master", "*.png")):
            frame = int(file.split("/")[-1].split(".")[0].split("_")[1])
            rgb_path.append([sequence, frame, file])

        rgb_path_df = pd.DataFrame(rgb_path, columns = ["Sequence", "Frame", "RGB"])
        rgb_path_df = rgb_path_df.sort_values("Frame", ignore_index=True)

        return rgb_path_df

    def _load_depth(self, current_path):
        """
        Description: Load the all depth data from .png to np.array(), concatenate all data
        into dataframe. Sort the dataframe according to the frame.

        Input:
        self: inherent from class
        current_path: indicated the current path of the data to be loaded.
        (Example): ../data/mmpose/train/sequence_9

        Output:
        Return a Tuple of format (sequence, frame, depth data)
        """

        depth_path = []
        sequence = int(current_path.split("/")[-1].split("_")[-1])
        for file in glob.glob(os.path.join(current_path, "depth", "master", "*.png")):
            frame = int(file.split("/")[-1].split(".")[0].split("_")[1])
            depth_path.append([sequence, frame, file])

        depth_path_df = pd.DataFrame(depth_path, columns = ["Sequence", "Frame", "Depth"])
        depth_path_df = depth_path_df.sort_values("Frame", ignore_index=True)

        return depth_path_df

    def _load_radar(self, current_path):
        """
        Description: Load the all radar data from .npy to np.array(), concatenate all data
        into dataframe. Sort the dataframe according to the frame.

        Input:
        self: inherent from class
        current_path: indicated the current path of the data to be loaded.
        (Example): ../data/mmpose/train/sequence_9

        Output:
        Return a Tuple of format (sequence, frame, radar data)
        """
        radar_path = []
        sequence = int(current_path.split("/")[-1].split("_")[-1])
        for file in glob.glob(os.path.join(current_path, "radar", "*.npy")):
            frame = int(file.split("/")[-1].split(".")[0].split("_")[1])
            radar_path.append([sequence, frame, file])
        radar_path_df = pd.DataFrame(radar_path, columns=["Sequence", "Frame", "Radar"])
        radar_path_df = radar_path_df.sort_values("Frame", ignore_index=True)

        return radar_path_df


    def _load_mesh(self, current_path):
        """
        Description: Load the all mesh data from .npy to np.array(), concatenate all data
        into dataframe. Sort the dataframe according to the frame.

        Input:
        self: inherent from class
        current_path: indicated the current path of the data to be loaded.
        (Example): ../data/mmpose/train/sequence_9

        Output:
        Return a Tuple of format (sequence, frame, mesh data)
        """
        mesh_path = []
        sequence = int(current_path.split("/")[-1].split("_")[-1])
        for file in glob.glob(os.path.join(current_path, "mesh", "*.npz")):
            frame = int(file.split("/")[-1].split(".")[0].split("_")[1])

            mesh_path.append([sequence, frame, file])
        mesh_path_df = pd.DataFrame(mesh_path,
                               columns=["Sequence", "Frame", "Mesh"])
        mesh_path_df = mesh_path_df.sort_values("Frame", ignore_index=True)

        return mesh_path_df



class MetaFi_Dataset(Dataset):
    r"""
    Implementation of "MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset".

    Point cloud Pose reconstruction dataset collected TI IWR6843 FWCM mmWave radar. Ground truth human pose are annotated in a self-supervised manner based on RGB images.
    Random split, cross-subject split and cross-environment split are supported. 

    Args:
        data_root (str of Path): Path to the dataset.

        split (str): Split of the dataset. Selected from ["training", "testing"] 

        config (Object): The configuration class object for the MetaFi dataset.

    Reference: 
        https://ntu-aiot-lab.github.io/mm-fi
    """

    def __init__(self, data_root, config, split = "training"):
        
        # database configuration

        self.data_root = data_root
        self.scenes = {}
        self.subjects = {}
        self.actions = {}
        self.modalities = {}
        self.load_database()

        # config decoding
        
        if split == "training":
            self.config_dict = self.decode_config(config)["train_dataset"]
        elif split == "testing":
            self.config_dict = self.decode_config(config)["val_dataset"]

        self.modality = self.config_dict["modality"].split('|')
        for m in self.modality:
            assert m in ['rgb', 'infra1', 'infra2', 'depth', 'lidar', 'mmwave', 'wifi-csi']  # 'rgb', 'infra1', 'infra2', 'depth',
        self.data_source = self.config_dict["data_form"]
        self.data_unit = config["data_unit"]
        self.split = split
        
        # load data

        self.data_list = self.load_data()


    def load_database(self):
        for scene in sorted(os.listdir(self.data_root)):
            self.scenes[scene] = {}
            for subject in sorted(os.listdir(os.path.join(self.data_root, scene))):
                self.scenes[scene][subject] = {}
                self.subjects[subject] = {}
                for action in sorted(os.listdir(os.path.join(self.data_root, scene, subject))):
                    self.scenes[scene][subject][action] = {}
                    self.subjects[subject][action] = {}
                    if action not in self.actions.keys():
                        self.actions[action] = {}
                    if scene not in self.actions[action].keys():
                        self.actions[action][scene] = {}
                    if subject not in self.actions[action][scene].keys():
                        self.actions[action][scene][subject] = {}
                    for modality in ['infra1', 'infra2', 'depth', 'rgb', 'lidar', 'mmwave', 'wifi-csi']:
                        data_path = os.path.join(self.data_root, scene, subject, action, modality)  # TODO: the path to the data file
                        self.scenes[scene][subject][action][modality] = data_path
                        self.subjects[subject][action][modality] = data_path
                        self.actions[action][scene][subject][modality] = data_path
                        if modality not in self.modalities.keys():
                            self.modalities[modality] = {}
                        if scene not in self.modalities[modality].keys():
                            self.modalities[modality][scene] = {}
                        if subject not in self.modalities[modality][scene].keys():
                            self.modalities[modality][scene][subject] = {}
                        if action not in self.modalities[modality][scene][subject].keys():
                            self.modalities[modality][scene][subject][action] = data_path


    def decode_config(self, config):
        all_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14',
                        'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28',
                        'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
        all_actions = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14',
                    'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27']
        train_form = {}
        val_form = {}
        # Limitation to actions (protocol)
        if config['protocol'] == 'protocol1':  # Daily actions
            actions = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
        elif config['protocol'] == 'protocol2':  # Rehabilitation actions:
            actions = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A15', 'A16', 'A24', 'A25', 'A26']
        else:
            actions = all_actions
        # Limitation to subjects and actions (split choices)
        if config['split_to_use'] == 'random_split':
            rs = config['random_split']['random_seed']
            ratio = config['random_split']['ratio']
            for action in actions:
                np.random.seed(rs)
                idx = np.random.permutation(len(all_subjects))
                idx_train = idx[:int(np.floor(ratio*len(all_subjects)))]
                idx_val = idx[int(np.floor(ratio*len(all_subjects))):]
                subjects_train = np.array(all_subjects)[idx_train].tolist()
                subjects_val = np.array(all_subjects)[idx_val].tolist()
                for subject in all_subjects:
                    if subject in subjects_train:
                        if subject in train_form:
                            train_form[subject].append(action)
                        else:
                            train_form[subject] = [action]
                    if subject in subjects_val:
                        if subject in val_form:
                            val_form[subject].append(action)
                        else:
                            val_form[subject] = [action]
                rs += 1
        elif config['split_to_use'] == 'cross_scene_split':
            subjects_train = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                            'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20',
                            'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']
            subjects_val = ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
            for subject in subjects_train:
                train_form[subject] = actions
            for subject in subjects_val:
                val_form[subject] = actions
        elif config['split_to_use'] == 'cross_subject_split':
            subjects_train = config['cross_subject_split']['train_dataset']['subjects']
            subjects_val = config['cross_subject_split']['val_dataset']['subjects']
            for subject in subjects_train:
                train_form[subject] = actions
            for subject in subjects_val:
                val_form[subject] = actions
        else:
            subjects_train = config['manual_split']['train_dataset']['subjects']
            subjects_val = config['manual_split']['val_dataset']['subjects']
            actions_train = config['manual_split']['train_dataset']['actions']
            actions_val = config['manual_split']['val_dataset']['actions']
            for subject in subjects_train:
                train_form[subject] = actions_train
            for subject in subjects_val:
                val_form[subject] = actions_val

        dataset_config = {'train_dataset': {'modality': config['modality'],
                                            'split': 'training',
                                            'data_form': train_form
                                            },
                        'val_dataset': {'modality': config['modality'],
                                        'split': 'validation',
                                        'data_form': val_form}}
        return dataset_config
            

    def get_scene(self, subject):
        if subject in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']:
            return 'E01'
        elif subject in ['S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']:
            return 'E02'
        elif subject in ['S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']:
            return 'E03'
        elif subject in ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']:
            return 'E04'
        else:
            raise ValueError('Subject does not exist in this dataset.')

    def load_data(self):
        data_info = tuple()
        for subject, actions in self.data_source.items():
            print(subject, actions)
            for action in actions:
                if self.data_unit == 'sequence':
                    data_dict = {'modality': self.modality,
                                 'scene': self.get_scene(subject),
                                 'subject': subject,
                                 'action': action,
                                 'gt_path': os.path.join(self.data_root, self.get_scene(subject), subject,
                                                         action, 'ground_truth.npy')
                                 }
                    for mod in self.modality:
                        data_dict[mod+'_path'] = os.path.join(self.data_root, self.get_scene(subject), subject,
                                                         action, mod)
                    data_info += (data_dict,)
                elif self.data_unit == 'frame':
                    
                    frame_list = sorted(os.listdir(os.path.join(self.data_root, self.get_scene(subject), subject, action, "mmwave")))
                    frame_num = len(frame_list)
                    for idx in range(frame_num):
                        frame_idx = int(frame_list[idx].split('.')[0].split('frame')[1]) - 1
                        data_dict = {'modality': self.modality,
                                        'scene': self.get_scene(subject),
                                        'subject': subject,
                                        'action': action,
                                        'gt_path': os.path.join(self.data_root, self.get_scene(subject), subject,
                                                                action, 'ground_truth.npy'),
                                        'idx': frame_idx
                                        }
                        for mod in self.modality:
                            data_dict[mod+'_path'] = os.path.join(self.data_root, self.get_scene(subject), subject, action, mod, sorted(os.listdir(os.path.join(self.data_root, self.get_scene(subject), subject, action, mod)))[idx])
                        data_info += (data_dict,)
                else:
                    raise ValueError('Unsupport data unit!')
        return data_info

    def read_dir(self, dir):
        _, mod = os.path.split(dir)
        data = []
        if mod in ['infra1', 'infra2', 'rgb']:
            for img in sorted(glob.glob(os.path.join(dir, "frame*.png"))):
                _cv_img = cv2.imread(img)  # Default is BGR color format
                data.append(_cv_img)
            data = np.array(data)
        elif mod == 'depth':
            for img in sorted(glob.glob(os.path.join(dir, "frame*.png"))):
                _cv_img = cv2.imread(img)  # Default is BGR color format
                data.append(_cv_img)
            data = np.array(data)
        elif mod == 'lidar':
            for bin_file in sorted(glob.glob(os.path.join(dir, "frame*.bin"))):
                with open(bin_file, 'rb') as f:
                    raw_data = f.read()
                    data_tmp = np.frombuffer(raw_data, dtype=np.float64)
                    data_tmp = data_tmp.reshape(-1, 3)
                data.append(data_tmp)
        elif mod == 'mmwave':
            for bin_file in sorted(glob.glob(os.path.join(dir, "frame*.bin"))):
                with open(bin_file, 'rb') as f:
                    raw_data = f.read()
                    data_tmp = np.frombuffer(raw_data, dtype=np.float64)
                    data_tmp = data_tmp.copy().reshape(-1, 5)
                    # data_tmp = data_tmp[:, :3]
                data.append(data_tmp)
        elif mod == 'wifi-csi':
            for csi_mat in sorted(glob.glob(os.path.join(dir, "frame*.mat"))):
                data_mat = scio.loadmat(csi_mat)['CSIamp']
                # data_frame = []
                # for i in range(data_mat.shape[2]):
                #     data_frame.append(data_mat[..., i].flatten())
                data_frame = np.array(data_mat)
                data.append(data_frame)
            data = np.array(data)
        else:
            raise ValueError('Found unseen modality in this dataset.')
        return data

    def read_frame(self, frame):
        _mod, _frame = os.path.split(frame)
        _, mod = os.path.split(_mod)
        if mod in ['infra1', 'infra2', 'rgb']:
            data = np.load(frame)
        elif mod == 'depth':
            data = cv2.imread(frame)  # TODO: 深度和RGB的读取格式好像有区别？我忘了，待定
        elif mod == 'lidar':
            with open(frame, 'rb') as f:
                raw_data = f.read()
                data = np.frombuffer(raw_data, dtype=np.float64)
                data = data.reshape(-1, 3)
        elif mod == 'mmwave':

            radar_path = frame
            frame_id = radar_path.split("/")[-1]
            frame = int(frame_id[5:8])
            radar_path_list = sliding_window(index_type="file", frames_together=5, sliding=1, input_path = radar_path, frame=frame, identifier="frame_")
            radar_data_list = []
            # current implentation
            for path in radar_path_list:

                with open(radar_path, 'rb') as f:
                    raw_data = f.read()
                radar_data = np.frombuffer(raw_data, dtype=np.float64)
                radar_data = radar_data.copy().reshape(-1, 5)
                radar_data = padding(radar_data, npoint=150)
                radar_data_list.append(radar_data)
            data = np.stack(radar_data_list, axis=0)


                # data = data[:, :3]
        elif mod == 'wifi-csi':
            data = scio.loadmat(frame)['CSIamp']
            data = np.array(data)
        else:
            raise ValueError('Found unseen modality in this dataset.')
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        r'''
        Returns:
            (input, label)[np.ndarray, int]: 
            "input" (np.ndarray): The point clouds from preprocessed radar point clouds. The shape of pc: [frame_together, npoints, 5], default [5, 150, 5].
            "label" (np.ndarray): The ground truth human pose. The shape of pose: [17, 3].

        '''
        item = self.data_list[idx]

        gt_numpy = np.load(item['gt_path'])
        gt_torch = torch.from_numpy(gt_numpy)

        if self.data_unit == 'sequence':
            sample = {'modality': item['modality'],
                      'scene': item['scene'],
                      'subject': item['subject'],
                      'action': item['action'],
                      'output': gt_torch
                      }
            for mod in item['modality']:
                data_path = item[mod+'_path']
                if os.path.isdir(data_path):
                    data_mod = self.read_dir(data_path)
                else:
                    data_mod = np.load(data_path + '.npy')
                sample['input_'+mod] = data_mod
        elif self.data_unit == 'frame':
            sample = {'modality': item['modality'],
                      'scene': item['scene'],
                      'subject': item['subject'],
                      'action': item['action'],
                      'idx': item['idx'],
                      'output': gt_torch[item['idx']]
                      }
            for mod in item['modality']:
                data_path = item[mod + '_path']
                if os.path.isfile(data_path):
                    data_mod = self.read_frame(data_path)
                    sample['input_'+mod] = data_mod
                else:
                    raise ValueError('{} is not a file!'.format(data_path))
        else:
            raise ValueError('Unsupport data unit!')
        return sample["input_mmwave"], sample["output"]





def load_hpe_dataset(dataset, root, config = None):
    r"""
    This function provide quick construct train-test dataset based on dataset name.

    Args:
        dataset (str): Name of dataset.

        dataset_root (str of path): Root dir of the data set.

    Return:
        train_dataset (torch.utils.data.Dataset): train split of dataset using pytorch.

        test_dataset (torch.utils.data.Dataset): test split of dataset using pytorch.

    """
    if dataset == 'mmBody':
        print('using dataset: mmBody DATA')
        train_dataset = MMBody(root, split='train', normalized=True,
                                     test_scenario="train", modalities=["Radar"])
        test_dataset = MMBody(root, split='test', normalized=True,
                                    test_scenario="all", modalities=["Radar"]) # can select lab1, lab2, rain...

    elif dataset == 'MetaFi':
        if config == None:
            print("Using default config file.") 
            with open('pysensing/mmwave/PC/dataset/hpe_config/MetaFi.yaml', 'r') as fd:
                config = yaml.load(fd, Loader=yaml.FullLoader)
        print('using dataset: MetaFi DATA')
        train_dataset = MetaFi_Dataset(root, config, split="training")
        test_dataset = MetaFi_Dataset(root, config, split="testing")


    else:
        raise ValueError("Unsupported dataset. Please choose from 'mmBody', 'MetaFi'.")

    return train_dataset, test_dataset