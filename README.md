# shift-attention

In AUTOMATIC1111/stable-diffusion-webui, generate a sequence of images shifting attention in the prompt.

This script enables you to give a range to the weight of tokens in a prompt and then generate a sequence of images stepping from the first one to the second.

The format is "(token:start value\~end value)" or ":start value\~end value" (please don't mix up the tilde "\~" for a minus sign "-"). Values can be anything you want, but usable values are probably around -1.5 to 1.5.

It will also allow multiple values, like this "(token:value1\~value2\~value3...\~valueN)". This can be used if you want the value to stay static for a while "(cat:0.1\~0.1\~1)" or jump between values "(dog:1\~0.5\~1\~-0.3)". The number of values do not have to be the same for every token in the prompt, the interpolation between them be spread out of the the number of steps you given.

One thing to keep in mind is that "(cat:0)" might still generate cats. If you want to morph from one thing into another you can use [Composable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#composable-diffusion) with the AND keyword. This example will morph from a cat into a dog:

cat :1\~0 AND dog :0\~1

A more advanced usage is to add the keyword "THEN" to make a list of prompts to shift. Example:

cat :1\~0 AND dog :0\~1 THEN dog THEN(seed=2) dog :1\~0 AND cat :0\~1

This would morph from "cat" to "dog", then shift "dog", slowly changing to seed 2 and lastly, morph from "dog" to "cat" again. If this feels confusing, don't worry, you do not have to use it if you. It also requires you to make sure the prompt before THEN will end in something that match the start of the prompt that comes after to get a smooth animation.

You can also "shift" the CFG scale. For example, if you set the CFG to something normal and then use this, you will get an animation of a cat that turns into something random:

cat THEN(cfg=0)

If the prompt and the negative prompt has different amount of prompts split up by THEN, the last prompt will be reused.

# Installation

The recommended way to install it is to:
1. Find "shift attention" in the list of available extensions in the webui, and then click Install.

If you can't find it in the list:
1. Go to the "Extensions" tab in the webui
2. Click on the "Install from URL" tab
3. Paste https://github.com/yownas/shift-attention.git into "URL for extension's git repository" and click install
4. ("Optional". You will need to restart the webui for dependensies to be installed or you won't be able to generate video files.)

Manual install (not recommmended):
1. Place the files from this repo in a folder in the extensions folder.
2. Restart. Pray. It might work properly. Maybe.

# Usage

`Steps (minimum)`: Number of steps to take from the initial image. The ranges you specified in the prompt will be spread out over these steps. "(minimum)" refers to SSIM usage (see below).

`FPS`: The Frames Per Second of the video. It has a hidden feature where if you set this to a negative value, it will be used as the length (in seconds) of the resulting video(s). Setting this to 0 will skip generating a video. 

`Lead in/out`: Amount of frames to be padded with a static image at the start and ending of the video. So you'll get a short pause before the video start/ends.

`SSIM threshold`: If this is set to something other than 0, the script will first generate the steps you've specified above, but then take a second pass and fill in the gaps between images that differ too much according to Structual Similarity Index Metric [(pdf)](https://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf). A good value depends a lot on which model and prompt you use, but 0.7 to 0.8 should be a good starting value. More than 0.95 will probably not improve much. If you want a very smooth video you should use something like [Flowframes](https://nmkd.itch.io/flowframes).

`SSIM CenterCrop%`: Crop a piece from the center of the image to be used for SSIM. In percent of the height and width. 0 will use the entire image. Only checking a small part of the image might make SSIM more sensitive. Be prepared to lower SSIM threshold to 0.4 to 0.5 if you use this.

`RIFE passes`: Use [Real-Time Intermediate Flow Estimation](https://github.com/vladmandic/rife) to interpolate between frames. Each pass will add 1 frame per frame, doubling the total number of frames. This does not change the fps above, so you need to keep that in mind if it is important to you. (This will save a seperate video file.)

`Drop original frames`: Drop the original frames and only keep the RIFE-frames. Keeping the same frame count and fps as before.

`Shift Attention extras`: Open this to show less used settings.

`Upscaler`: Choose upscale method to be applied to the images before made into a video.

`Upscale ratio`: How much the images should be upscaled. A value of 0 or 1 will disable scaling.

`Show generated images in ui`: Disable this if you generate a lot of steps to make life easier for your browser.

`Mirror mode`: Let the animation "bounce back" to the start by playing it in reverse.

`SSIM minimum step`: Smallest "step" SSIM is allowed to take. Sometimes animations can't me smoothed out, no matter how small steps you take. It is better to let the script give up and have a single skip than force it and get an animation that flickers.

`SSIM min threshold`: Try to make new images "at least" this good. By default SSIM will give up a newly generated image is worse then the gap it is trying to fill. This will allow you to set "Steps" to something as low as 1 and not have SSIM give up just because the image halfway through was bad.

# Show your work

If you want to you can send me a link (with prompt) and I'll add it to [USER_EXAMPLES.md](USER_EXAMPLES.md).

# Example:

Prompt: "photo of (cat:1\~0) or (dog:0\~1)"

![shift_attention](https://user-images.githubusercontent.com/13150150/193368922-be51b5b8-7d8a-4499-b089-64dd7112b9d3.png)

# Result:

https://user-images.githubusercontent.com/13150150/193368939-c0a57440-1955-417c-898a-ccd102e207a5.mp4
