
import SimpleITK as sitk
import glob
import os
import pandas as pd

main_folder = "/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente/data//TAVI/CT_postprocessed"
file_names = ['0f7f6475.nii.gz']
destination_path = "/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente/data//TAVI/CT_postprocessed"
os.makedirs(destination_path, exist_ok=True)
for file_path in file_names:
    img = sitk.ReadImage(os.path.join(main_folder, file_path))
    img = sitk.DICOMOrient(img, 'RAS')
    sitk.WriteImage(img, os.path.join(destination_path, file_path))
    print(file_path)


#import nibabel as nib
#img = nib.load(r"C:\Users\Federica\Desktop\RedBrickData\Federica_CT-Aorta\CT-aorta_segmentations\Dossier_No1_COMPLETE-4d006b-TDM-STD00001-SER00005\Dossier_No1_COMPLETE-4d006b-TDM-STD00001-SER00002.nii.gz")
#print('\naffine :',img.affine)

#img = nib.as_closest_canonical(img)
# check orientation of the 3D CT
#x, y, z = nib.aff2axcodes(img.affine)
#x,y,z = img.shape
#print('\naffine after conversion:',img.affine)
#print('\nImage orientation:', x, y, z)

#ras_i = x - 274
#ras_j = y - 241
#ras_k = 413
#print(ras_i, ras_j, ras_k)