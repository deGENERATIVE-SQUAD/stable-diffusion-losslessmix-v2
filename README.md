# stable-diffusion-losslessmix-v2
This is a simplified fork of https://github.com/recoilme/losslessmix
# What improved
1. You can use from 2 models to an infinity (you can mix any quantity models you want, but if you mix 2 models - it is just a Weighted sum 0.5 (use alpha/beta args then)).
2. Script calculates average tensor values from all models you choose, in my research it is much better than default choosing existing values near the base model because much more smoothie. 
# How to use
**python losslessmixv2.py --models** model1.ckpt model2.ckpt model3.ckpt **--out name**

You can use these args:
1. --maxdiff (Maxdiff algo from https://www.reddit.com/r/StableDiffusion/comments/1012lto/comment/j7aoyso/?context=3 )
2. --alpha (e.g. --alpha 0.9)
3. --beta (e.g. --beta 1.1)

# Recommendations
Recommend to use with basic SD/NAI models in A position but you can experiment ofcourse.
# Changelog
+ v2.2
(in progress) Harmonic Mean and Geometric mean implementation
+ v2.1 
Fixed KeyErrors when key is not present in one of the models being merged.
+ v2
Initial commit
# Future updates
Testing another types of calculation which more powerful than average values
