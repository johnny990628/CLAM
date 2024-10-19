import h5py
import numpy as np
import matplotlib.pyplot as plt

class WSIPatchReader:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.file = None

    def __enter__(self):
        self.file = h5py.File(self.h5_path, 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def get_top_level_keys(self):
        return list(self.file.keys())

    def is_group(self, key):
        return isinstance(self.file[key], h5py.Group)

    def is_dataset(self, key):
        return isinstance(self.file[key], h5py.Dataset)

    def get_group_keys(self, group_name):
        if self.is_group(group_name):
            return list(self.file[group_name].keys())
        return None

    def get_dataset(self, path):
        if path in self.file:
            return np.array(self.file[path])
        return None

    def get_attribute(self, path, attr_name):
        if path in self.file and attr_name in self.file[path].attrs:
            return self.file[path].attrs[attr_name]
        return None

    def display_random_patch(self, group_name):
        if self.is_group(group_name) and 'coords' in self.file[group_name] and 'patches' in self.file[group_name]:
            patches = np.array(self.file[group_name]['patches'])
            idx = np.random.randint(0, patches.shape[0])
            plt.imshow(patches[idx])
            plt.title(f"Random Patch from {group_name}")
            plt.axis('off')
            plt.show()
        else:
            print(f"Cannot display patch. Required datasets not found in {group_name}")

# Usage example
if __name__ == "__main__":
    h5_file_path = "/work/u6658716/TCGA-LUAD/CLAM/FEATURES_40x_20x/h5_files/TCGA-4B-A93V-01Z-00-DX1.C263DC1C-298D-47ED-AAF8-128043828530.h5"

    with WSIPatchReader(h5_file_path) as reader:
        top_level_keys = reader.get_top_level_keys()
        print(f"Top-level keys: {top_level_keys}")

        for key in top_level_keys:
            if reader.is_group(key):
                print(f"\nExploring group: {key}")
                group_keys = reader.get_group_keys(key)
                if group_keys:
                    print(f"Keys in {key}: {group_keys}")
                    for subkey in group_keys:
                        dataset = reader.get_dataset(f"{key}/{subkey}")
                        if dataset is not None:
                            print(f"  {subkey} shape: {dataset.shape}")
                    
                    # Try to display a random patch if possible
                    reader.display_random_patch(key)
                else:
                    print(f"{key} is an empty group")
            elif reader.is_dataset(key):
                print(f"\n{key} is a dataset")
                dataset = reader.get_dataset(key)
                print(f"Shape: {dataset.shape}")
                print(f"Data type: {dataset.dtype}")
                if dataset.size < 10:  # Print small datasets entirely
                    print(f"Data: {dataset}")
                else:  # Print first few elements of larger datasets
                    print(f"First few elements: {dataset.flat[:5]}...")
            else:
                print(f"\n{key} is neither a group nor a dataset")

        # You can add more specific data retrievals or analyses here
