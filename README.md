# stable-diffusion-losslessmix-v2
This is a simplified fork of https://github.com/recoilme/losslessmix

# Script versions

1. Average Sum in losslessmixv2.1.py:
uses **avg_value = sum(values) / num_models**
2. Weighted Mean in losslessmixv2.1wm.py: uses **weighted_avg_value = sum(weighted_values)**
3. Bayesian approach in losslessmixv2.1bav2.py: uses Bayesian approach with --max_posterior

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

# Recommendations
+ Recommend to use with basic SD/NAI models in A position but you can experiment ofcourse.
+ Also recommend to use --alpha 0.0 argument to apply A model text encoder without halving values and merging of each halfs of a text encoder from models (like Weighted Sum algo does) but you can experiment with an alpha/beta mixing like from original script.

# Changelog
+ v2.2
(in progress)
+ v2.1wm as Weighted mean tensor mixing function implementation
+ v2.1 as 
Fixed KeyErrors when key is not present in one of the models being merged.
+ v2
Initial commit
# Future updates
Testing another types of calculation which more powerful than average values
