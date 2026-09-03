import nibabel as nib
import pandas as pd
import os
import numpy as np
import trimesh
from skimage import measure
import open3d as o3d
from totalsegmentator.python_api import totalsegmentator

main_folder = "/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente/data//TAVI/CT_postprocessed"
file_names = os.listdir(main_folder)
destination_path = "/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente/data//TAVI/validation_landmarks"
os.makedirs(destination_path, exist_ok=True)
num_list = [50]

for name in file_names:
    for num in num_list:
        subject = os.path.splitext(os.path.splitext(name)[0])[0]

        print(f'Subject: {subject}')
        input_path = os.path.join(main_folder, name)
        output_path = os.path.join(destination_path, name)
        if __name__ == "__main__":
            vertebrae = totalsegmentator(
                input_path,
                output_path,
                roi_subset=[
                    'vertebrae_T12', 'vertebrae_T11', 'vertebrae_T10', 'vertebrae_T9', 'vertebrae_T8', 'vertebrae_T7',
                    'vertebrae_T6', 'vertebrae_T5', 'vertebrae_T4', 'vertebrae_T3', 'vertebrae_T2', 'vertebrae_T1',

                ],
                ml=True
            )

            aortic_sinuses = totalsegmentator(
                input_path,
                output_path,
                task="aortic_sinuses",
                ml=True
            )
            
            
            aorta = totalsegmentator(
                input_path,
                output_path,
                roi_subset=[
                    'aorta',

                ],                ml=True
            )

            # --- cut the aorta above the top of the aortic sinuses ---
            affine = aorta.affine
            aorta_data = aorta.get_fdata()
            sinuses_data = aortic_sinuses.get_fdata()

            sin_vox = np.argwhere(sinuses_data > 0)
            sin_world = (np.c_[sin_vox, np.ones(len(sin_vox))] @ affine.T)[:, :3]
            s_cut = sin_world[:, 2].min()  # min of sinuses in superior (S) axis

            aorta_vox = np.argwhere(aorta_data > 0)
            aorta_world = (np.c_[aorta_vox, np.ones(len(aorta_vox))] @ affine.T)[:, :3]
            keep = aorta_world[:, 2] > s_cut

            aorta_cut = np.zeros_like(aorta_data, dtype=np.uint8)
            vi = aorta_vox[keep]
            aorta_cut[vi[:, 0], vi[:, 1], vi[:, 2]] = 1

            # --- group all segmentations into one img ---
            combined = np.zeros_like(aorta_data, dtype=np.uint8)
            combined[vertebrae.get_fdata() > 0] = 1
            # combined[sinuses_data > 0] = 1
            # combined[aorta_cut > 0] = 1
            img = nib.Nifti1Image(combined.astype(np.uint8), affine, aorta.header)
            nib.save(img, output_path)
            img_data = img.get_fdata()
            binary_mask = (img_data > 0).astype(np.uint8)

            verts, faces, _, _ = measure.marching_cubes(binary_mask, level=0.5)
            verts_h = np.c_[verts, np.ones(len(verts))]
            verts_world = verts_h @ affine.T
            verts_world = verts_world[:, :3]
            mesh = trimesh.Trimesh(vertices=verts_world, faces=faces)
            components = mesh.split(only_watertight=False)
            mesh = max(components, key=lambda m: m.area)
            #mesh.export( os.path.join(destination_path, subject + ".stl"))

            mesh_o3d = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(mesh.vertices),
                o3d.utility.Vector3iVector(mesh.faces)
            )


            pcd_dense = mesh_o3d.sample_points_uniformly(number_of_points=5000)
            points = pcd_dense.farthest_point_down_sample(num_samples  =num)

            df = pd.DataFrame(points.points, columns=["r", "a", "s"])
            os.makedirs(destination_path, exist_ok=True)
            csv_file_path = os.path.join(destination_path, subject + f'_{num}_toracic.csv')
            print(csv_file_path)
            df.to_csv(csv_file_path, index=False)