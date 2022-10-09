# shift-attention

In AUTOMATIC1111/stable-diffusion-webui, generate a sequence of images shifting attention in the prompt.

This script enables you to give a range to the weight of tokens in a prompt and then generate a sequence of images stepping from the first one to the second.

The format is "(token:start value\~end value)" or ":start value\~end value" (please don't mix up the tilde "\~" for a minus sign "-"). Values can be anything you want, but usable values are probably around -1.5 to 1.5.

# Installation
1. Copy the file in the scripts-folder to the scripts-folder from https://github.com/AUTOMATIC1111/stable-diffusion-webui
2. Add `moviepy==1.0.3` to requirements_versions.txt

# Show your work

If you want to you can send me a link (with prompt) and I'll add it to [USER_EXAMPLES.md](USER_EXAMPLES.md).

# Example:

Prompt: "photo of (cat:1\~0) or (dog:0\~1)"

![shift_attention](https://user-images.githubusercontent.com/13150150/193368922-be51b5b8-7d8a-4499-b089-64dd7112b9d3.png)

# Result:

https://user-images.githubusercontent.com/13150150/193368939-c0a57440-1955-417c-898a-ccd102e207a5.mp4
