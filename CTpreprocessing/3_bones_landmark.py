from totalsegmentator.python_api import totalsegmentator
#import nibabel as nib
import pandas as pd
import os
import numpy as np

main_folder = '/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente/data/TAVI/CT_postprocessed'
file_names = ["0f7f6475.nii.gz"]
destination_path = "/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente/data/TAVI/isocenter"
os.makedirs(destination_path, exist_ok=True)

for name in file_names:
    data_list = []
    input_path = os.path.join(main_folder, name)
    output_path = os.path.join(destination_path, name)
    dest_name = os.path.splitext(os.path.splitext(name)[0])[0]
    print(dest_name)
    if __name__ == "__main__":
        img = totalsegmentator(input_path, output_path, roi_subset=['vertebrae_T8'], ml = True)
        img_data = img.get_fdata()
        affine = img.affine
        img_data = np.where(img_data >0, 1, 0)

        indices = np.argwhere(img_data == 1)

        if len(indices) == 0:
            data_list.append({
                'path': name,
                'T8x': 0,
                'T8y': 0,
                'T8z': 0
            })
            raise SystemError()

        # Compute the mean of the indices (centroid calculation)
        centroid = np.round(np.mean(indices, axis=0))
        print(centroid)
        data_list.append({
            'path': name,
            'T8x': centroid[0],
            'T8y': centroid[1],
            'T8z': centroid[2]
        })

        #new_img = nib.Nifti1Image(img_data, affine)
        #nib.save(new_img, output_path)

        df = pd.DataFrame(data_list)

        # Save DataFrame to CSV file
        csv_file_path = os.path.join(destination_path, dest_name + '.csv')
        df.to_csv(csv_file_path, index=False)