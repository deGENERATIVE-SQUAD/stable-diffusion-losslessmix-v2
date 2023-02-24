# stable-diffusion-losslessmix-v2
This is a simplified fork of https://github.com/recoilme/losslessmix

# Script versions

1. Average Sum in losslessmixv2.1.py:
uses **avg_value = sum(values) / num_models**
2. Weighted Mean in losslessmixv2.1wm.py: uses **weighted_avg_value = sum(weighted_values)**
3. Bayesian approach in losslessmixv2.1bav2.py: **uses Bayesian approach with --max_posterior**
4. Cosine similarity in losslessmixv2.1cs.py: it is like from new version https://github.com/recoilme/losslessmix but can mix multiple models

# What improved
1. You can use from 2 models to an infinity (you can mix any quantity models you want).
2. Script calculates average tensor values from all models you choose, in my research it is much better than default choosing existing values near the base model because much more smoothie. 
# How to use

1. For losslessmixv2.1.py and losslessmixv2.1wm.py: 
   + **python losslessmixv2.1.py --models** model1.ckpt model2.ckpt model3.ckpt **--out name**

   You can use these args:
   + --maxdiff (Maxdiff algo from https://www.reddit.com/r/StableDiffusion/comments/1012lto/comment/j7aoyso/?context=3 )
   + --alpha (e.g. --alpha 0.9)
   + --beta (e.g. --beta 1.1)

2. For losslessmixv2.1bav2.py:
   + python losslessmixv2.1bav2.py --models 1.ckpt 2.ckpt 3.ckp  --out name --max_posterior
 
2. For losslessmixv2.1cs.py:
   + python losslessmixv2.1cs.py model1.ckpt model2.ckpt model3.ckpt --out name

   You can use these args:
   + --soft for soft cosine similarity mode (sometimes broken with multiple models, need to fix)
   + --s arg not tested
   
# Recommendations
In basic losslessmixv2.1.py and losslessmixv2.1wm.py versions:
+ Recommend to use with basic SD/NAI models in A position but you can experiment ofcourse.
+ Also recommend to use --alpha 0.0 argument to apply A model text encoder without halving values and merging of each halfs of a text encoder from models (like Weighted Sum algo does) but you can experiment with an alpha/beta mixing like from original script.

In losslessmixv2.1bav2.py version:
+ Bayesian approach can not into --alpha and --beta, do not use it.

# Changelog
+ v2.2
(in progress)
+ 2.1cs as https://github.com/recoilme/losslessmix with multiple models mixing
+ v2.1bav2 as Bayesian approach implementation
+ v2.1wm as Weighted mean tensor mixing function implementation
+ v2.1 as 
Fixed KeyErrors when key is not present in one of the models being merged.
+ v2
Initial commit
# Future updates
Testing another types of calculation which more powerful than average values

# Examples

All images upscaled through Ultimate Upscale from 512 to 1024 + Valar upscale model, DS 0.37

Base gen using CFG scale fix (CFG 14 -> mimic 7, Half Cosine Up, 7, Half Cosine Up, 7, 0)

Model set in one merge: SD 1.5 + NAI + AOM2 + grapefruitv4 + babes11 + corneos7thHeavenMix_v2 + hyperass + CounterfeitV25

Pos: woman, sfw

Neg: worst quality, low quality, deformed, distorted, disfigured poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, monochrome, greyscale

**+ losslessmixv2.1cs --soft**

![This is an image](https://i.imgur.com/ONe8jG2.png)

**+ losslessmixv2.1bav2 --max-posterior**

![This is an image](https://i.imgur.com/eoIADa9.png)

