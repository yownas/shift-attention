# User examples

Feel free to post a link with animations you've done and want to share, preferably with prompt and other settings as an Issue or PR and I'll add them here. (Unless they are too explicit or include gore or other controversial topics. Please keep it clean.)

# Yownas

https://www.youtube.com/watch?v=ohMv2OxCKO4

street view of a (old medieval village with derelict wood houses:0.3\~0) or ((dystopian) bright neon cyberpunk city:0.7\~1.4), (at daytime:0.9\~0) or (at night:0.2\~1.2), epic scale, cinema view, raytraced, realistic digital illustration

Negative prompt: text, watermark, day, dusk, medieval, town, cyberpunk, city, people, cartoon, pixelart

Sampling method: DDIM Sampling Steps: 20 CFG Scale: 12

Seed: 2295097780

Script: Shift attention (https://github.com/yownas/shift-atten...)

Steps: 450 FPS: 15

---

Same prompt, 10 initial steps and SSIM threshold: 0.975. This is probably as high as you can go with this prompt and the old SD1.4 model. The animation will stutter when it tries to make too small changes, not sure if it is a floatingpoint error or simply the model that reach a point where even a very small change will result in a "big" change in the image.

https://user-images.githubusercontent.com/13150150/222804476-06ea4014-3fab-41ce-9540-6c09a561bf59.mp4

