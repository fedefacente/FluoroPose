from totalsegmentator.python_api import totalsegmentator
import nibabel as nib
import pandas as pd
import os
import numpy as np


main_folder = "/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente/data//TAVI/raw_CT/raw_cropped_CT"
file_names = os.listdir(main_folder)
destination_path = "/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente/data//TAVI/CT_postprocessed"
os.makedirs(destination_path, exist_ok=True)

for name in file_names:
    print(name)
    input_path = os.path.join(main_folder, name)
    print(input_path)
    output_path = os.path.join(destination_path,  'segm.nii.gz')
    if __name__ == "__main__":
        img = totalsegmentator(
            input_path,
            output_path,
            roi_subset=[
                'aorta', 
                'heart',
                'pulmonary_vein',
                'brachiocephalic_trunk',
                'subclavian_artery_right',
                'subclavian_artery_left',
                'common_carotid_artery_right',
                'common_carotid_artery_left',
                'brachiocephalic_vein_left',
                'brachiocephalic_vein_right',
                'atrial_appendage_left',
                'superior_vena_cava',
                'inferior_vena_cava',
                'portal_vein_and_splenic_vein',
                'iliac_artery_left',
                'iliac_artery_right',
                'iliac_vena_left',
                'iliac_vena_right'
            ],
            ml=True
        )
        img_data = img.get_fdata()
        affine = img.affine
        img_data = np.where(img_data >0, 1, 0)

        ct_img = nib.load(input_path)
        ct_data = ct_img.get_fdata()
        affine = ct_img.affine
        ct_data = np.where(img_data == 1, 40,ct_data )

        nifti_img = nib.Nifti1Image(ct_data, affine)
        output_file = os.path.join(destination_path,name)
        nib.save(nifti_img, output_file)



