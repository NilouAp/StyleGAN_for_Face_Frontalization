import os
import torch
from tqdm import tqdm
import shutil



class FacePairDatasetPreparer:
    """
    A class to prepare profile and frontal image pairs for identity-preserving face frontalization.

    Attributes:
        images_dir (str): Path to the root directory containing the face images.
        profile_dest (str): Destination directory for profile images.
        frontal_dest (str): Destination directory for frontal images.
        camera_exclusions (list): List of camera angles to exclude.
        ids_list (list): Sorted list of all IDs to process.
    """

    def __init__(self, images_dir, profile_dest, frontal_dest, camera_exclusions, ids_file):
        """
        Initializes the FacePairDatasetPreparer.

        Args:
            images_dir (str): Root directory containing face images.
            profile_dest (str): Directory to save profile images.
            frontal_dest (str): Directory to save frontal images.
            camera_exclusions (list): List of cameras to exclude.
            ids_file (str): Path to the pre-saved list of sorted IDs.
        """
        self.images_dir = images_dir
        self.profile_dest = profile_dest
        self.frontal_dest = frontal_dest
        self.camera_exclusions = camera_exclusions
        self.ids_list = torch.load(ids_file)
        self.image_pair_count = 0

    def prepare_pairs(self, max_ids=200):
        """
        Prepares profile-frontal image pairs for training.

        Args:
            max_ids (int): Maximum number of IDs to process.
        """
        for i in tqdm(range(min(max_ids, len(self.ids_list))), desc="Processing IDs"):
            exp_folds_sorted = self.load_sorted_list(f"exp_folds_sorted.pt")
            for exp_fold in exp_folds_sorted:
                pose_folds_sorted = self.load_sorted_list(f"pose_folds_sorted.pt")
                self.process_poses(exp_fold, pose_folds_sorted)

    def process_poses(self, exp_fold, pose_folds_sorted):
        """
        Processes poses for a given expression folder.

        Args:
            exp_fold (str): Current expression folder name.
            pose_folds_sorted (list): Sorted list of pose folders.
        """
        for pose_fold in pose_folds_sorted:
            if pose_fold not in self.camera_exclusions:
                self.process_images(exp_fold, pose_fold)

    def process_images(self, exp_fold, pose_fold):
        """
        Processes images within a pose folder.

        Args:
            exp_fold (str): Current expression folder name.
            pose_fold (str): Current pose folder name.
        """
        image_dir = os.path.join(self.images_dir, self.ids_list[self.image_pair_count], exp_fold, pose_fold)
        image_files = sorted(os.listdir(image_dir))
        for image_file in image_files:
            self.save_image_pair(image_file, exp_fold, pose_fold)

    def save_image_pair(self, image_file, exp_fold, pose_fold):
        """
        Saves the profile and corresponding frontal image pair.

        Args:
            image_file (str): Name of the profile image file.
            exp_fold (str): Current expression folder name.
            pose_fold (str): Current pose folder name.
        """
        if 'Thumbs.db' in image_file:
            return

        # Save profile image
        profile_image_path = os.path.join(self.images_dir, self.ids_list[self.image_pair_count], exp_fold, pose_fold, image_file)
        profile_destination = os.path.join(self.profile_dest, str(self.image_pair_count))
        shutil.copy(profile_image_path, profile_destination)

        # Generate corresponding frontal image name
        token = image_file.split('_')
        frontal_image_name = f"{token[0]}_01_{token[2]}_051_07.png"
        frontal_image_path = os.path.join(
            self.images_dir,
            self.ids_list[self.image_pair_count],
            exp_fold,
            '05_1',
            frontal_image_name
        )
        frontal_destination = os.path.join(self.frontal_dest, str(self.image_pair_count))

        # Save frontal image
        shutil.copy(frontal_image_path, frontal_destination)
        self.image_pair_count += 1

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
    Main function to prepare profile-frontal image pairs for training.
    """
    IMAGES_DIR = '/home/niloufar/Desktop/Niloufar/CMU-PIE/session01/multiview'
    PROFILE_DEST = '/home/niloufar/Desktop/Niloufar/CMU_pairs_for_psp/profiles'
    FRONTAL_DEST = '/home/niloufar/Desktop/Niloufar/CMU_pairs_for_psp/frontals'
    CAMERA_EXCLUSIONS = ['19_1', '08_1']
    IDS_FILE = '/home/niloufar/Desktop/Niloufar/stylegan2-ada-pytorch/All_IDs_sorted.pt'

    dataset_preparer = FacePairDatasetPreparer(
        images_dir=IMAGES_DIR,
        profile_dest=PROFILE_DEST,
        frontal_dest=FRONTAL_DEST,
        camera_exclusions=CAMERA_EXCLUSIONS,
        ids_file=IDS_FILE
    )

    dataset_preparer.prepare_pairs(max_ids=200)


if __name__ == "__main__":
    main()
