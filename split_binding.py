import os
import random


SPLIT_FILENAMES = {
    "train": "train_files.txt",
    "val": "val_files.txt",
    "test": "test_files.txt",
}


def split_dataset_counts(total_len):
    if total_len <= 0:
        return 0, 0, 0
    if total_len == 1:
        return 1, 0, 0
    if total_len == 2:
        return 1, 1, 0

    train_count = max(1, int(0.6 * total_len))
    val_count = max(1, int(0.2 * total_len))
    test_count = total_len - train_count - val_count

    if test_count <= 0:
        if train_count > val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            train_count = max(1, train_count - 1)
        test_count = total_len - train_count - val_count

    return train_count, val_count, test_count


def limit_training_dataset(train_dataset, train_size=None):
    dataset = list(train_dataset)
    if train_size is None:
        return dataset

    train_size = int(train_size)
    if train_size <= 0:
        raise ValueError(f"train_size must be positive, got {train_size}")
    if train_size > len(dataset):
        raise ValueError(
            f"train_size={train_size} exceeds available training graphs ({len(dataset)})"
        )
    return dataset[:train_size]


def save_split_files(results_dir, train_dataset, val_dataset, test_dataset):
    split_datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    split_paths = {}

    for split_name, dataset in split_datasets.items():
        split_path = os.path.join(results_dir, SPLIT_FILENAMES[split_name])
        with open(split_path, "w", encoding="utf-8") as handle:
            for data in dataset:
                handle.write(data.filename + "\n")
        split_paths[split_name] = split_path

    return split_paths


def deterministic_split_dataset(full_dataset, seed):
    shuffled_dataset = list(full_dataset)
    random.Random(seed).shuffle(shuffled_dataset)

    train_count, val_count, test_count = split_dataset_counts(len(shuffled_dataset))
    train_end = train_count
    val_end = train_count + val_count

    return {
        "train": shuffled_dataset[:train_end],
        "val": shuffled_dataset[train_end:val_end],
        "test": shuffled_dataset[val_end:val_end + test_count],
    }


def load_split_filenames(pretrain_path):
    model_dir = os.path.dirname(pretrain_path)
    split_paths = {}
    split_filenames = {}

    for split_name, split_filename in SPLIT_FILENAMES.items():
        split_path = os.path.join(model_dir, split_filename)
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"Strict split binding requires {split_path}, but it was not found."
            )
        with open(split_path, "r", encoding="utf-8") as handle:
            split_filenames[split_name] = [line.strip() for line in handle if line.strip()]
        split_paths[split_name] = split_path

    return split_paths, split_filenames


def bind_dataset_to_split_files(full_dataset, pretrain_path):
    split_paths, split_filenames = load_split_filenames(pretrain_path)
    data_by_filename = {}

    for data in full_dataset:
        if data.filename in data_by_filename:
            raise ValueError(f"Duplicate graph filename in dataset: {data.filename}")
        data_by_filename[data.filename] = data

    split_datasets = {}
    assigned = set()
    for split_name in ("train", "val", "test"):
        filenames = split_filenames[split_name]
        overlap = assigned.intersection(filenames)
        if overlap:
            raise ValueError(
                f"Strict split binding found overlapping filenames across split manifests: "
                + ", ".join(sorted(overlap)[:10])
                + (" ..." if len(overlap) > 10 else "")
            )

        missing = [name for name in filenames if name not in data_by_filename]
        if missing:
            raise ValueError(
                f"Strict split binding references files missing from the current dataset in {split_name}: "
                + ", ".join(missing[:10])
                + (" ..." if len(missing) > 10 else "")
            )

        split_datasets[split_name] = [data_by_filename[name] for name in filenames]
        assigned.update(filenames)

    if not split_datasets["train"]:
        raise ValueError("Strict split binding produced an empty training set.")
    if not split_datasets["val"]:
        raise ValueError("Strict split binding produced an empty validation set.")
    if not split_datasets["test"]:
        raise ValueError("Strict split binding produced an empty test set.")

    return split_datasets, split_paths
