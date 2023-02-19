# stable-diffusion-losslessmix-v2
This is a simplified fork of https://github.com/recoilme/losslessmix
# What improved
1. You can use from 2 models to an infinity (you can mix any quantity models you want).
2. Script calculates average tensor values from all models you choose, in my research it is much better than default choosing existing values near the base model because much more smoothie. 
# How to use
**python losslessmixv2.py --models** model1.ckpt model2.ckpt model3.ckpt **--out name**
# Recommendations
Recommend to use with basic SD/NAI models in A position but you can experiment ofcourse.
# Future updates
Testing another types of calculation which more powerful than average values
