import os
import pandas as pd

COCO_DIR = "train2014"
SYNTHETIC_DIR = "coco_synthetic"
TRAIN_FILE = "train_filter.txt"
TEST_FILE = "test_filter.txt"

train_df = pd.read_csv(TRAIN_FILE, delimiter=" ", header=None)
test_df = pd.read_csv(TEST_FILE, delimiter=" ", header=None)

all_files = set(list(train_df[0].values) + list(test_df[0].values))

for filename in os.listdir(SYNTHETIC_DIR):
    if filename not in all_files:
        os.remove(os.path.join(SYNTHETIC_DIR, filename))
        print(filename + "removed")

for filename in os.listdir(COCO_DIR):
    if filename not in all_files:
        os.remove(os.path.join(COCO_DIR, filename))
        print(filename + "removed")
