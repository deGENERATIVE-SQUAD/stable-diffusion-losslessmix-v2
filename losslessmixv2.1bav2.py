import argparse
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Merge models with max diff")
parser.add_argument("--models", type=str, nargs='+', help="Paths to models", required=True)
parser.add_argument("--out", type=str, help="Output file name, without extension", default="merged")
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu")
parser.add_argument("--without_vae", action="store_false", help="Do not merge VAE")
parser.add_argument("--dry", action="store_true", help="dry_run", default=False)
parser.add_argument("--max_posterior", action="store_true", help="use max posterior", default=False)
args = parser.parse_args()

def load_model_weights(mPath):
    model = torch.load(mPath, map_location=args.device).get("state_dict", torch.load(mPath, map_location=args.device))
    return model

output_file = f'{args.out}.ckpt'
models = [load_model_weights(mPath) for mPath in args.models]

a = models[0]
for key in tqdm(a.keys(), desc="Stage 1/3"):
    if args.without_vae and "first_stage_model" in key: continue
    if "model" in key:
        values = [m.get(key) for m in models]
        if None in values: continue
        values = [v.float() for v in values]  # convert non-None values to float

        # Compute Bayesian posterior by treating values as samples from the posterior distribution
        mean_value = torch.mean(torch.stack(values), dim=0)
        std_value = torch.std(torch.stack(values), dim=0)

        if args.max_posterior:
            # Set a[key] to the value with the maximum posterior probability
            a[key] = mean_value + std_value * torch.sign(mean_value - values[0])
        else:
            # Set a[key] to the mean of the posterior distribution
            a[key] = mean_value


for m in models[1:]:
    for key in tqdm(m.keys(), desc="Stage 2/3"):
        if "model" in key and key not in a:
            a[key] = m[key]

if not args.dry:
    print("Saving...")
    torch.save({"state_dict": a}, output_file)

print("Done!")
