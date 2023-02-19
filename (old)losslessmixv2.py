import os
import argparse
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Merge models with max diff")
parser.add_argument("--models", type=str, nargs='+', help="Paths to models", required=True)
parser.add_argument("--out", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
parser.add_argument("--without_vae", action="store_false", help="Do not merge VAE", required=False)
parser.add_argument("--dry", action="store_true", help="dry_run", default=False, required=False)
parser.add_argument("--alpha", type=float, help="multiply models by alpha", default=1.0, required=False)
parser.add_argument("--beta", type=float, help="multiply models by beta", default=1.0, required=False)
parser.add_argument("--maxdiff", action="store_true", help="use max diff", default=False, required=False)

args = parser.parse_args()


def loadModelWeights(mPath):
	model = torch.load(mPath, map_location=args.device)
	try: theta = model["state_dict"]
	except: theta = model
	return theta


output_file = f'{args.out}.ckpt'

step = 0
models = [loadModelWeights(mPath) for mPath in args.models]
num_models = len(models)
a = models[0]

for key in tqdm(a.keys(), desc="Stage 1/3"):
    # skip VAE model parameters to get better results
    if args.without_vae and "first_stage_model" in key: continue
    if "model" in key:
        values = [m[key] for m in models]
        avg_value = sum(values) / num_models
        if step == 1 or step == 2:
            print(f'step:{step}')
            print(f'a:{1000*a[key]}')
            for i, value in enumerate(values):
                print(f'{args.models[i]}: {1000*value}')
            print(f'avg: {1000*avg_value}')
        step += 1
        if args.maxdiff:
            a[key] = avg_value * (abs(a[key] - args.alpha*avg_value) > abs(a[key] - args.beta*values[0])) + values[0] * (abs(a[key] - args.alpha*avg_value) <= abs(a[key] - args.beta*values[0]))
        else:
            a[key] = values[-1] * (abs(a[key] - args.alpha*avg_value) > abs(a[key] - args.beta*values[-1])) + avg_value * (abs(a[key] - args.alpha*avg_value) <= abs(a[key] - args.beta*values[-1]))

for m in models[1:]:
    for key in tqdm(m.keys(), desc="Stage 2/3"):
        if "model" in key and key not in a:
            a[key] = m[key]

if args.dry == 0:
    print("Saving...")
    torch.save({"state_dict": a}, output_file)

print("Done!")
