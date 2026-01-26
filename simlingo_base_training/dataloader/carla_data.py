"""
Code that loads the dataset for training.
partially taken from https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py
(MIT licence)
"""

import numpy as np
import random
import cv2

from simlingo_base_training.dataloader.dataset_base import BaseDataset

VIZ_DATA = False

class CARLA_Data(BaseDataset):  # pylint: disable=locally-disabled, invalid-name
    """
    Custom dataset that dynamically loads a CARLA dataset from disk.
    """

    def __init__(self,
            **cfg,
        ):
        super().__init__(**cfg, base=True)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        # Disable threading because the data loader will already split in threads.
        cv2.setNumThreads(0)
        
        # mh 20260125: Retry logic for handling corrupted images/files
        max_retries = 3  # Increased retries to handle multiple corrupted samples
        original_index = index
        tried_indices = {index}
        
        for retry in range(max_retries):
            try:
                return self._load_sample(index)
            except (FileNotFoundError, ValueError) as e:
                # If it's an image loading error, try a different sample
                if retry < max_retries - 1:
                    # Try random index to avoid consecutive bad samples
                    # Keep trying until we find an index we haven't tried yet
                    import random
                    for _ in range(100):  # Try up to 100 times to find a new index
                        new_index = random.randint(0, len(self.images) - 1)
                        if new_index not in tried_indices:
                            index = new_index
                            tried_indices.add(index)
                            break
                    else:
                        # If we've tried too many indices, just increment
                        index = (index + 1) % len(self.images)
                        tried_indices.add(index)
                    continue
                else:
                    # If all retries failed, raise the error with more context
                    raise ValueError(
                        f"Failed to load sample after {max_retries} retries. "
                        f"Original index: {original_index}, tried indices: {sorted(tried_indices)[:10]}. "
                        f"Last error: {e}"
                    )
            except Exception as e:
                # For other errors, don't retry
                raise e
    
    def _load_sample(self, index):
        """Internal method to load a single sample."""
        data = {}
        images = self.images[index]
        boxes = self.boxes[index]
        measurements = self.measurements[index]
        sample_start = self.sample_start[index]
        augment_exists = self.augment_exists[index]

        ######################################################
        ######## load current and future measurements ########
        ######################################################
        # mh 20260120 
        # 从 measurements/ 加载 JSON.gz 文件，包含：
        # 车辆速度
        # 目标点 (target_point, target_point_next)
        # 命令 (command, next_command)
        # 其他传感器数据

        loaded_measurements, current_measurement, measurement_file_current = self.load_current_and_future_measurements(
            measurements,
            sample_start
            )
        
        data['measurement_path'] = measurement_file_current

        # Determine whether the augmented camera or the normal camera is used.
        if augment_exists and random.random() <= self.img_shift_augmentation_prob and self.img_shift_augmentation:
            augment_sample = True
            aug_rotation = current_measurement['augmentation_rotation']
            aug_translation = current_measurement['augmentation_translation']
        else:
            augment_sample = False
            aug_rotation = 0.0
            aug_translation = 0.0

        data['augment_sample'] = augment_sample
        data['aug_rotation'] = aug_rotation
        data['aug_translation'] = aug_translation


        ######################################################
        ################## load waypoints ####################
        ## mh 20260120 加载waypoints 从测量数据提取，作为训练标签（Ground Truth）
        ######################################################
        data = self.load_waypoints(data, loaded_measurements, aug_translation, aug_rotation)
       
        data['speed'] = current_measurement['speed']

        data = self.load_route(data, current_measurement, aug_translation, aug_rotation)

        target_point = np.array(current_measurement['target_point'])
        target_point = self.augment_target_point(target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
        next_target_point = np.array(current_measurement['target_point_next'])
        next_target_point = self.augment_target_point(next_target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
        
        data['target_point'] = target_point
        data['next_target_point'] = next_target_point
        ######################################################
        ######## load current and past images ########
        ######################################################
        data = self.load_images(data, images, augment_sample=augment_sample)
        # print(f"########## self.route_as is: {self.route_as} ##########")   #### mh 20260120 target_point
        if self.route_as == 'coords':
            map_route = data['route'][:20]
            data['map_route'] = map_route
        elif self.route_as == 'target_point':
            tp = [target_point, next_target_point]
            tp = np.array(tp)
            data['map_route'] = tp
        else:
            raise ValueError(f"Unknown route_as: {self.route_as}")
            #### ########## self.route_as is: target_point ##########
########## data['map_route'] is: [[ 3.66759424e+01 -2.24885232e-03][ 5.72388075e+01 -3.69690201e-03]]
###data keys are: dict_keys(['measurement_path', 'augment_sample', 'aug_rotation', 'aug_translation', 'waypoints', 'waypoints_org', 'waypoints_1d', 'ego_waypoints', 'ego_waypoints_org', 'speed', 'route', 'route_adjusted_org', 'route_adjusted', 'target_point', 'next_target_point', 'rgb', 'rgb_org_size', 'map_route']) ##########
        # print(f"########## data['map_route'] is: {data['map_route']} ##########")
        # print(f"########## data keys are: {data.keys()} ##########")
        # print(f"*****waypoints is: {data['waypoints']} ##########")
        return data


if __name__ == "__main__":
    from hydra import compose, initialize

    initialize(config_path="../config")
    cfg = compose(config_name="config")

    print('Test Dataset')
    dataset = CARLA_Data(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        data_path=cfg.dataset.data_path,
        bucket_path=cfg.dataset.bucket_path,
        hist_len=cfg.dataset.hist_len,
        pred_len=cfg.dataset.pred_len,
        skip_first_n_frames=cfg.dataset.skip_first_n_frames,
        num_route_points=cfg.dataset.num_route_points,
        split="train",
        bucket_name="all",
    )

    for i in range(len(dataset)):
        data = dataset[i]
        print(data)
        if i == 10:
            break
