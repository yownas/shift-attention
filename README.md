# shift-attention

In AUTOMATIC1111/stable-diffusion-webui, generate a sequence of images shifting attention in the prompt.

This script enables you to give a range to the weight of tokens in a prompt and then generate a sequence of images stepping from the first one to the second.

The format is "(token:start value\~end value)" or ":start value\~end value" (please don't mix up the tilde "\~" for a minus sign "-"). Values can be anything you want, but usable values are probably around -1.5 to 1.5.

It will also allow multiple values, like this "(token:value1\~value2\~value3...\~valueN)". This can be used if you want the value to stay static for a while "(cat:0.1\~0.1\~1)" or jump between values "(dog:1\~0.5\~1\~-0.3)". The number of values do not have to be the same for every token in the prompt, the interpolation between them be spread out of the the number of steps you given.

# Installation

Easiest way to install it is to:
1. Go to the "Extensions" tab in the webui
2. Click on the "Install from URL" tab
3. Paste https://github.com/yownas/shift-attention.git into "URL for extension's git repository" and click install
4. ("Optional". You will need to restart the webui for dependensies to be installed or you won't be able to generate video files.)

Manual install:
1. Copy the file in the scripts-folder to the scripts-folder from https://github.com/AUTOMATIC1111/stable-diffusion-webui
2. Add `moviepy==1.0.3` to requirements_versions.txt

# Show your work

If you want to you can send me a link (with prompt) and I'll add it to [USER_EXAMPLES.md](USER_EXAMPLES.md).

# Example:

Prompt: "photo of (cat:1\~0) or (dog:0\~1)"

![shift_attention](https://user-images.githubusercontent.com/13150150/193368922-be51b5b8-7d8a-4499-b089-64dd7112b9d3.png)

# Result:

https://user-images.githubusercontent.com/13150150/193368939-c0a57440-1955-417c-898a-ccd102e207a5.mp4
