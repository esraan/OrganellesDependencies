from DModel import *
import os
import pandas as pd
from torch.utils.data import DataLoader
def split_data(images_dir="/storage/users/g-and-n/plates/metadata/", metadata_dir="/storage/users/g-and-n/plates/images/",plates=["24278","24277"]):
    total_ctrl_metadata = pd.DataFrame()
    metadata_plate_paths = metadata_plates_gen(images_dir,plates,metadata_dir)
    total_df = pd.DataFrame()
    for metadata_plate in metadata_plate_paths:
        curr_df = pd.read_csv(metadata_plate)
        curr_df["main_path"] = metadata_plate.rstrip(".csv")+"/"
        total_df = pd.concat([total_df, curr_df[curr_df["Well_Role"] == "mock"]])
    return total_df
        
        
        
    # mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
    print("g")

def metadata_plates_gen(images_dir,plates,metadata_dir):
    metadata_plate_paths = [os.path.join(images_dir, plate + ".csv") for plate in sorted(os.listdir(metadata_dir)) if
                            plate in plates]
    for plate_path in metadata_plate_paths:
        yield plate_path
        
def image_path_mock(image):
    pass

# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

df = split_data()
print("here!!!!")