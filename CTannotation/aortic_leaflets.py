from totalsegmentator.python_api import totalsegmentator




from totalsegmentator.python_api import totalsegmentator
import nibabel as nib
import pandas as pd
import os
import numpy as np
main_folder = "/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente/data//TAVI/CT_postprocessed"
file_names = os.listdir(main_folder)
destination_path = "/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente/data//TAVI/CT_segm_annotations_leaflets"
os.makedirs(destination_path, exist_ok=True)

for name in file_names:
    print(name)
    input_path = os.path.join(main_folder, name)
    output_path = os.path.join(destination_path, name)
    if __name__ == "__main__":
        img= totalsegmentator(
            input_path,
            output_path,
            task="aortic_sinuses",
            ml=True,          # single multilabel .nii.gz instead of one mask per label
        )
        
        # img_data = img.get_fdata()
        # affine = img.affine
        # # img_data = np.where(img_data ==3, 1, 0).astype(np.uint8)
        
        # new_img = nib.Nifti1Image(img_data, affine)
        # new_img.set_data_dtype(np.uint8)
        # nib.save(new_img, output_path)