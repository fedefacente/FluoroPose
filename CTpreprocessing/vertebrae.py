from totalsegmentator.python_api import totalsegmentator
import nibabel as nib
import pandas as pd
import os
import numpy as np

main_folder = "/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente/data//TAVI/CT_postprocessed"
file_names = os.listdir(main_folder)
destination_path = "/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente/data//TAVI/CT_segm_annotations_vertebrae"
os.makedirs(destination_path, exist_ok=True)

for name in file_names:
    print(name)
    input_path = os.path.join(main_folder, name)
    print(input_path)
    output_path = os.path.join(destination_path,  name)
    if __name__ == "__main__":
        img = totalsegmentator(
            input_path,
            output_path,
            roi_subset=[
                'vertebrae_S1', 'vertebrae_L5', 'vertebrae_L4', 'vertebrae_L3', 'vertebrae_L2', 'vertebrae_L1',
                'vertebrae_T12', 'vertebrae_T11', 'vertebrae_T10', 'vertebrae_T9', 'vertebrae_T8', 'vertebrae_T7',
                'vertebrae_T6', 'vertebrae_T5', 'vertebrae_T4', 'vertebrae_T3', 'vertebrae_T2', 'vertebrae_T1',
                'vertebrae_C7', 'vertebrae_C6', 'vertebrae_C5', 'vertebrae_C4', 'vertebrae_C3', 'vertebrae_C2',
                'vertebrae_C1', 
            ],
            ml=True
        )
        img_data = img.get_fdata()
        affine = img.affine
        img_data = np.where(img_data >0, 1, 0).astype(np.uint8)
        
        new_img = nib.Nifti1Image(img_data, affine)
        new_img.set_data_dtype(np.uint8)
        nib.save(new_img, output_path)