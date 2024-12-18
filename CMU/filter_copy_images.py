import os
import torch
from tqdm import tqdm
import shutil


class DatasetFilterCopier:
    """
    A class to filter and copy images based on specific criteria, excluding unwanted camera views.

    Attributes:
        images_dir (str): Path to the directory containing the image dataset.
        destination_dir (str): Path to the destination directory for filtered images.
        excluded_cameras (list): List of camera views to exclude.
        ids_list (list): Sorted list of IDs for processing.
    """

    def __init__(self, images_dir, destination_dir, excluded_cameras, ids_file):
        """
        Initializes the DatasetFilterCopier.

        Args:
            images_dir (str): Root directory containing image datasets.
            destination_dir (str): Directory to save filtered images.
            excluded_cameras (list): List of camera views to exclude.
            ids_file (str): Path to the pre-saved sorted list of IDs.
        """
        self.images_dir = images_dir
        self.destination_dir = destination_dir
        self.excluded_cameras = excluded_cameras
        self.ids_list = torch.load(ids_file)

    def filter_and_copy_images(self, max_ids=200):
        """
        Filters and copies images from the dataset to the destination directory.

        Args:
            max_ids (int): Maximum number of IDs to process.
        """
        for i in tqdm(range(min(max_ids, len(self.ids_list))), desc="Processing IDs"):
            exp_folds_sorted = self.load_sorted_list('exp_folds_sorted.pt')
            for exp_fold in exp_folds_sorted:
                pose_folds_sorted = self.load_sorted_list('pose_folds_sorted.pt')
                self.process_poses(exp_fold, pose_folds_sorted)

    def process_poses(self, exp_fold, pose_folds_sorted):
        """
        Processes pose folders for a given expression folder.

        Args:
            exp_fold (str): Current expression folder name.
            pose_folds_sorted (list): Sorted list of pose folders.
        """
        for pose_fold in pose_folds_sorted:
            if pose_fold not in self.excluded_cameras:
                self.copy_images(exp_fold, pose_fold)

    def copy_images(self, exp_fold, pose_fold):
        """
        Copies images from a pose folder to the destination directory.

        Args:
            exp_fold (str): Current expression folder name.
            pose_fold (str): Current pose folder name.
        """
        pose_folder_path = os.path.join(self.images_dir, self.ids_list[self.image_index], exp_fold, pose_fold)
        image_files = sorted(os.listdir(pose_folder_path))

        for image_file in image_files:
            source_path = os.path.join(pose_folder_path, image_file)
            destination_path = os.path.join(self.destination_dir, image_file)
            shutil.copy(source_path, destination_path)

    @staticmethod
    def load_sorted_list(filepath):
        """
        Loads a pre-saved sorted list.

        Args:
            filepath (str): Path to the sorted list file.

        Returns:
            list: Loaded sorted list.
        """
        return torch.load(filepath)



def main():
    """
    Main function to filter and copy dataset images.
    """
    IMAGES_DIR = '/home/niloufar/Desktop/Niloufar/CMU-PIE/session01/multiview'
    DESTINATION_DIR = '/home/niloufar/Desktop/Niloufar/CMU_pairs_for_psp/all_exept75_90'
    EXCLUDED_CAMERAS = ['19_1', '08_1', '01_0', '24_0', '12_0', '11_0']
    IDS_FILE = '/home/niloufar/Desktop/Niloufar/stylegan2-ada-pytorch/All_IDs_sorted.pt'

    dataset_copier = DatasetFilterCopier(
        images_dir=IMAGES_DIR,
        destination_dir=DESTINATION_DIR,
        excluded_cameras=EXCLUDED_CAMERAS,
        ids_file=IDS_FILE
    )

    dataset_copier.filter_and_copy_images(max_ids=200)


if __name__ == "__main__":
    main()
