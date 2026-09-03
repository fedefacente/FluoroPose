from totalsegmentator.python_api import totalsegmentator
import nibabel as nib
import pandas as pd
import os
import numpy as np
import trimesh
from skimage import measure
import open3d as o3d
main_folder = "/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente/data//TAVI/CT_postprocessed"
file_names = os.listdir(main_folder)
destination_path = "/srv/storage/epione@storage2.sophia.grid5000.fr/ffacente/data//TAVI/CT_segm_annotations"
os.makedirs(destination_path, exist_ok=True)
num_list = [14,50,100,500,1000,5000]

for name in file_names:
    for num in num_list:
        print(name)
        subject = os.path.splitext(os.path.splitext(name)[0])[0]

        input_path = os.path.join(main_folder, name)
        output_path = os.path.join(destination_path, name)
        if __name__ == "__main__":
            img = totalsegmentator(
                input_path,
                output_path,
                roi_subset=[
                    'vertebrae_S1', 'vertebrae_L5', 'vertebrae_L4', 'vertebrae_L3', 'vertebrae_L2', 'vertebrae_L1',
                    'vertebrae_T12', 'vertebrae_T11', 'vertebrae_T10', 'vertebrae_T9', 'vertebrae_T8', 'vertebrae_T7',
                    'vertebrae_T6', 'vertebrae_T5', 'vertebrae_T4', 'vertebrae_T3', 'vertebrae_T2', 'vertebrae_T1',
                    'vertebrae_C7', 'vertebrae_C6', 'vertebrae_C5', 'vertebrae_C4', 'vertebrae_C3', 'vertebrae_C2',
                    'vertebrae_C1'
                ],
                ml=True
            )

            img_data = img.get_fdata()
            affine = img.affine
            img_data = np.where(img_data >0, 1, 0)
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
            csv_file_path = os.path.join(destination_path, subject + f'_{num}.csv')
            print(csv_file_path)
            df.to_csv(csv_file_path, index=False)