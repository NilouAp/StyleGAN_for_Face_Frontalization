import splitfolders

def split_dataset(input_dir, output_dir, seed=1337, ratio=(0.98, 0.02), group_prefix=None, move=False):
    """
    Splits a dataset into training and validation sets based on the specified ratio.

    Args:
        input_dir (str): Path to the directory containing all the data to be split.
        output_dir (str): Path to the directory where the split data will be saved.
        seed (int, optional): Seed for reproducibility. Defaults to 1337.
        ratio (tuple, optional): Ratio to split the data (train, validation). Defaults to (0.98, 0.02).
        group_prefix (str, optional): Prefix to group files by folder. Defaults to None.
        move (bool, optional): Whether to move files instead of copying. Defaults to False.

    Returns:
        None
    """
    splitfolders.ratio(
        input=input_dir,
        output=output_dir,
        seed=seed,
        ratio=ratio,
        group_prefix=group_prefix,
        move=move
    )
    print(f"Dataset split completed:\n- Input: {input_dir}\n- Output: {output_dir}\n- Split Ratio: {ratio}")


def main():
    """
    Main function to execute dataset splitting with specified parameters.
    """
    # Define paths and parameters
    input_directory = "/home/niloufar/Desktop/Niloufar/CMU_pairs_for_psp/all-data-dlib-256-CMU"
    output_directory = "/home/niloufar/Desktop/Niloufar/CMU_pairs_for_psp/split-data-dlib-256-CMU"
    split_seed = 1337
    split_ratio = (0.98, 0.02)

    # Perform the dataset split
    split_dataset(
        input_dir=input_directory,
        output_dir=output_directory,
        seed=split_seed,
        ratio=split_ratio,
        group_prefix=None,
        move=False
    )


if __name__ == "__main__":
    main()
