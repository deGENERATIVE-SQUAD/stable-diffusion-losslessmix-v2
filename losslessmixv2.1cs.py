import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Merge models with weighted similarity")
parser.add_argument("models", type=str, nargs='+', help="Paths to models")
parser.add_argument("--out", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
parser.add_argument("--without_vae", action="store_false", help="Do not merge VAE", required=False)
parser.add_argument("--dry", action="store_true", help="dry_run", default=False, required=False)
parser.add_argument("--soft", action="store_true", help="soft", default=False, required=False)
parser.add_argument("--s", type=float, help="share a b", default=.5, required=False)
args = parser.parse_args()


def loadModelWeights(mPaths):
    theta = []
    for mPath in mPaths:
        model = torch.load(mPath, map_location=args.device)
        try: 
            theta.append(model["state_dict"])
        except: 
            theta.append(model)
    return theta


output_file = f'{args.out}-{args.s}.ckpt'

step = 0
thetas = loadModelWeights(args.models)
sim = torch.nn.CosineSimilarity(dim=0)
sims = np.array([], dtype=np.float64)

for i in range(len(thetas)-1):
    a, b = thetas[i], thetas[i+1]
    for key in tqdm(a.keys(), desc=f"Stage {i+1}/{len(thetas)*2-2}"):
        # skip VAE model parameters to get better results
        if args.without_vae and "first_stage_model" in key: continue
        if "model" in key and key in b:
            simab = sim(a[key].to(torch.float64), b[key].to(torch.float64))
            sims = np.append(sims,simab.numpy())
sims = sims[~np.isnan(sims)]
if sims.size == 0:
    print('sims array is empty')
if sims.size > 0:
    sims = np.delete(sims, np.where(sims<np.percentile(sims, 1)), axis=0)
if len(sims) > 0:
    sims = sims[~np.isnan(sims)]
    if len(sims) > 0 and len(np.where(sims<np.percentile(sims, 1))[0]) > 0:
        sims = np.delete(sims, np.where(sims<np.percentile(sims, 1))[0], axis=0)
    if len(sims) > 0 and len(np.where(sims>np.percentile(sims, 99))[0]) > 0:
        sims = np.delete(sims, np.where(sims>np.percentile(sims, 99))[0], axis=0)
    if len(sims) == 0:
        print('Empty sims')
    else:
        print(len(sims),sims.min(),sims.max())
else:
    print('Empty sims')

a = thetas[0]
for i in range(len(thetas)-1):
    b = thetas[i+1]
    j = -1
    for key in tqdm(a.keys(), desc=f"Stage {i+len(thetas)}/{len(thetas)*2-2}"):
        # skip VAE model parameters to get better results
        if args.without_vae and "first_stage_model" in key: continue
        if "model" in key and key in b:
            j += 1
            simab = sim(a[key].to(torch.float32), b[key].to(torch.float32))
            k = (simab - sims.min())/(sims.max() - sims.min())
            k = k - args.s
            k = k.clip(min=.0,max=1.)
            if args.soft:
                a[key] = a[key] * (1 - k) + b[key] * k
            else:
                a[key] = a[key] * k + b[key] * (1 - k)
            a[key] = a[key].to(torch.float16)
        if i == len(thetas)-2:
            if key not in b: 
                a[key] = a[key].to(torch.float16)
            else: 
                a[key] = b[key].to(torch.float16)

if args.dry == 0:
    print("Saving...")
    torch.save({"state_dict": a}, output_file)

print("Done!")
