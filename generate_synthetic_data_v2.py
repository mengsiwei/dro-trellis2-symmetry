"""
Generate synthetic training data using TRELLIS.2
Based on DSO's generate_synthetic_data.py
Changes: trellis->trellis2, return_latent for sparse_x0/cond, o_voxel GLB export
"""
from glob import glob
import os
import argparse
from PIL import Image
import torch
from tqdm import tqdm

os.environ['SPCONV_ALGO'] = 'native'

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

TRELLIS2_MODEL_PATH = "/root/autodl-tmp/TRELLIS.2-4B"

parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=int, default=0)
parser.add_argument("--num_jobs", type=int, default=1)
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--save_extra", action="store_true")
parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/dso-data")
parser.add_argument("--image_paths", type=str, required=True)
parser.add_argument("--pipeline_type", type=str, default="512")
args = parser.parse_args()

pipeline = Trellis2ImageTo3DPipeline.from_pretrained(TRELLIS2_MODEL_PATH)
pipeline.cuda()

sample_batch_size, num_batches = 1, args.num_samples

if isinstance(args.image_paths, str):
    images = sorted(glob(args.image_paths))
else:
    images = sorted(args.image_paths)

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
print(f"################ Total images: {len(images)} ################")

for image_path in tqdm(images[args.job_id::args.num_jobs]):
    image = Image.open(image_path)
    image = pipeline.preprocess_image(image)

    category, prompt, image_name = image_path.split("/")[-3:]
    save_root = os.path.join(output_dir, f"{category}/{prompt}")
    os.makedirs(save_root, exist_ok=True)

    if len(glob(os.path.join(save_root, f"{image_name.replace('.jpg', '_*.glb').replace('.png', '_*.glb')}"))) >= args.num_samples:
        continue

    for bid in range(num_batches):
        skip_batch = True
        for i in range(sample_batch_size):
            eid = bid * sample_batch_size + i
            glb_path = os.path.join(save_root, f"{image_name.replace('.jpg', f'_{eid:03d}.glb').replace('.png', f'_{eid:03d}.glb')}")
            sparse_path = os.path.join(save_root, f"{image_name.replace('.jpg', f'_sparse_sample_{eid:03d}.pt').replace('.png', f'_sparse_sample_{eid:03d}.pt')}")
            if not os.path.exists(glb_path) or not os.path.exists(sparse_path):
                skip_batch = False
                break
        if skip_batch:
            continue

        try:
            # return_latent=True gives: (meshes, cond_dict, sparse_x0, (shape_slat, tex_slat, res))
            meshes, cond, sparse_x0, latent_tuple = pipeline.run(
                image, seed=bid, num_samples=sample_batch_size,
                preprocess_image=False, return_latent=True,
                pipeline_type=args.pipeline_type,
                sparse_structure_sampler_params={"steps": 12, "guidance_strength": 7.5},
                shape_slat_sampler_params={"steps": 12, "guidance_strength": 7.5},
            )
        except Exception as e:
            print(f"Failed for {image_path}: {e}")
            continue

        _, _, res = latent_tuple

        if args.save_extra:
            if bid == 0:
                image_cond = cond["cond"][0].to(dtype=torch.bfloat16)
                cond_path = os.path.join(save_root, f"{image_name.replace('.jpg', '_cond.pt').replace('.png', '_cond.pt')}")
                torch.save(image_cond.cpu(), cond_path)

            for i in range(sample_batch_size):
                eid = bid * sample_batch_size + i
                sparse_sample = sparse_x0[i].to(dtype=torch.bfloat16)
                sparse_path = os.path.join(save_root, f"{image_name.replace('.jpg', f'_sparse_sample_{eid:03d}.pt').replace('.png', f'_sparse_sample_{eid:03d}.pt')}")
                torch.save(sparse_sample.cpu(), sparse_path)

        for i in range(min(sample_batch_size, len(meshes))):
            eid = bid * sample_batch_size + i
            glb_path = os.path.join(save_root, f"{image_name.replace('.jpg', f'_{eid:03d}.glb').replace('.png', f'_{eid:03d}.glb')}")
            try:
                mesh = meshes[i]
                glb = o_voxel.postprocess.to_glb(
                    vertices=mesh.vertices, faces=mesh.faces,
                    attr_volume=mesh.attrs, coords=mesh.coords,
                    attr_layout=pipeline.pbr_attr_layout,
                    grid_size=res,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=100000, texture_size=1024,
                )
                glb.export(glb_path)
            except Exception as e:
                print(f"GLB export failed for {glb_path}: {e}")
                # Fallback: trimesh with geometry only (sufficient for physics sim)
                try:
                    import trimesh
                    tm = trimesh.Trimesh(
                        vertices=mesh.vertices.cpu().numpy(),
                        faces=mesh.faces.cpu().numpy())
                    tm.export(glb_path)
                except Exception as e2:
                    print(f"Trimesh fallback failed: {e2}")
