# Shift Attention script for AUTOMATIC1111/stable-diffusion-webui
#
# https://github.com/yownas/shift-attention
#
# Give a prompt like: "photo of (cat:1~0) or (dog:0~1)"
# Generates a sequence of images, lowering the weight of "cat" from 1 to 0 and increasing the weight of "dog" from 0 to 1.
# Will also support multiple numbers. "(cat:1~0~1)" will go from cat:1 to cat:0 to cat:1 streched over the number of steps

import os
import re
import modules.scripts as scripts
import gradio as gr
import math
import numpy
from torchmetrics import StructuralSimilarityIndexMeasure 
from torchvision import transforms
import torch
import random
from modules.processing import Processed, process_images, fix_seed
from modules.shared import opts, cmd_opts, state, sd_upscalers
from modules.images import resize_image

__ = lambda key, value=None: opts.data.get(f'customscript/seed_travel.py/txt2img/{key}/value', value)

DEFAULT_UPSCALE_METH   = __('Upscaler', 'Lanczos')
DEFAULT_UPSCALE_RATIO  = __('Upscale ratio', 1.0)
CHOICES_UPSCALER  = [x.name for x in sd_upscalers]

class Script(scripts.Script):
    def title(self):
        return "Shift attention"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        steps = gr.Number(label='Steps (minimum)', value=10)
        with gr.Row():
            show_images = gr.Checkbox(label='Show generated images in ui', value=True)
            ssim_diff = gr.Slider(label='SSIM threshold (1.0=exact copy)', value=0.0, minimum=0.0, maximum=1.0, step=0.01)
        with gr.Row():
            save_video = gr.Checkbox(label='Save results as video', value=True)
            ssim_ccrop = gr.Slider(label='SSIM CenterCrop% (0 to disable)', value=0, minimum=0, maximum=100, step=1)
        with gr.Row():
            video_fps = gr.Number(label='Frames per second', value=30)
            lead_inout = gr.Number(label='Number of frames for lead in/out', value=0)
        with gr.Row():
            upscale_meth  = gr.Dropdown(label='Upscaler',    value=lambda: DEFAULT_UPSCALE_METH, choices=CHOICES_UPSCALER)
            upscale_ratio = gr.Slider(label='Upscale ratio', value=lambda: DEFAULT_UPSCALE_RATIO, minimum=0.0, maximum=8.0, step=0.1)
        substep_min = gr.Number(label='SSIM minimum step', value=0.0001)

        return [steps, save_video, video_fps, show_images, lead_inout, upscale_meth, upscale_ratio, ssim_diff, ssim_ccrop, substep_min]

    def get_next_sequence_number(path):
        from pathlib import Path
        """
        Determines and returns the next sequence number to use when saving an image in the specified directory.
        The sequence starts at 0.
        """
        result = -1
        dir = Path(path)
        for file in dir.iterdir():
            if not file.is_dir(): continue
            try:
                num = int(file.name)
                if num > result: result = num
            except ValueError:
                pass
        return result + 1

    def run(self, p, steps, save_video, video_fps, show_images, lead_inout, upscale_meth, upscale_ratio, ssim_diff, ssim_ccrop, substep_min):
        re_attention_span = re.compile(r"([\-.\d]+~[\-~.\d]+)", re.X)

        def shift_attention(text, distance):

            def inject_value(distance, match_obj):
                a = match_obj.group(1).split('~')
                l = len(a) - 1
                q1 = int(math.floor(distance*l))
                q2 = int(math.ceil(distance*l))
                return str( float(a[q1]) + ((float(a[q2]) - float(a[q1])) * (distance * l - q1)) )

            res = re.sub(re_attention_span, lambda match_obj: inject_value(distance, match_obj), text)
            return res

        initial_info = None
        images = []
        dists = []
        lead_inout = int(lead_inout)
        tgt_w, tgt_h = round(p.width * upscale_ratio), round(p.height * upscale_ratio)

        if not save_video and not show_images:
            print(f"Nothing to do. You should save the results as a video or show the generated images.")
            return Processed(p, images, p.seed)

        if save_video:
            import numpy as np
            try:
                import moviepy.video.io.ImageSequenceClip as ImageSequenceClip
            except ImportError:
                print(f"moviepy python module not installed. Will not be able to generate video.")
                return Processed(p, images, p.seed)

        # Custom folder for saving images/animations
        shift_path = os.path.join(p.outpath_samples, "shift")
        os.makedirs(shift_path, exist_ok=True)
        shift_number = Script.get_next_sequence_number(shift_path)
        shift_path = os.path.join(shift_path, f"{shift_number:05}")
        p.outpath_samples = shift_path
        if save_video: os.makedirs(shift_path, exist_ok=True)

        # Force Batch Count and Batch Size to 1.
        p.n_iter = 1
        p.batch_size = 1

        # Make sure seed is fixed
        fix_seed(p)

        total_images = int(steps)
        print(f"Generating {total_images} images.")

        # Set generation helpers
        state.job_count = total_images

        initial_prompt = p.prompt
        initial_negative_prompt = p.negative_prompt

        # Generate all the steps
        for i in range(int(steps) + 1):
            if state.interrupted:
                break

            distance = float(i / int(steps))
            p.prompt = shift_attention(initial_prompt, distance)
            p.negative_prompt = shift_attention(initial_negative_prompt, distance)

            proc = process_images(p)
            if initial_info is None:
                initial_info = proc.info

            # upscale - copied from https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel
            if upscale_meth != 'None' and upscale_ratio != 1.0 and upscale_ratio != 0.0:
                image = [resize_image(0, proc.images[0], tgt_w, tgt_h, upscaler_name=upscale_meth)]
            else:
                image = [proc.images[0]]

            images += image
            dists += [distance]

        # SSIM
        if ssim_diff > 0:
            ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
            if ssim_ccrop == 0:
                transform = transforms.Compose([transforms.ToTensor()])
            else:
                transform = transforms.Compose([transforms.CenterCrop(min(tgt_w, tgt_h)*(ssim_ccrop/100)), transforms.ToTensor()])

            transform = transforms.Compose([transforms.ToTensor()])

            check = True
            skip_count = 0
            skip_ssim_min = 1.0

            done = 0
            while(check):
                if state.interrupted:
                    break
                check = False
                for i in range(done, len(images)-1):
                    # Check distance between i and i+1

                    a = transform(images[i].convert('RGB')).unsqueeze(0)
                    b = transform(images[i+1].convert('RGB')).unsqueeze(0)
                    d = ssim(a, b)

                    if d < ssim_diff and (dists[i+1] - dists[i]) > substep_min:
                        # FIXME debug output
                        print(f"SSIM: {dists[i]} <-> {dists[i+1]} = ({dists[i+1] - dists[i]}) {d}")

                        # Add image and run check again
                        check = True
                        new_dist = (dists[i] + dists[i+1])/2.0

                        p.prompt = shift_attention(initial_prompt, new_dist)
                        p.negative_prompt = shift_attention(initial_negative_prompt, new_dist)

                        print(f"Process: {new_dist}")
                        proc = process_images(p)

                        if initial_info is None:
                            initial_info = proc.info
                        
                        # upscale - copied from https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel
                        if upscale_meth != 'None' and upscale_ratio != 1.0 and upscale_ratio != 0.0:
                            image = [resize_image(0, proc.images[0], tgt_w, tgt_h, upscaler_name=upscale_meth)]
                        else:
                            image = [proc.images[0]]

                        #images = images[0:i] + [image] + images[i:]
                        images.insert(i+1, image[0])
                        dists.insert(i+1, new_dist)
                        break;
                    else:
                        # DEBUG
                        if d > ssim_diff:
                            if i > done:
                                print(f"Done: {dists[i+1]*100}% ({d}) {len(dists)} frames.   ")
                        else:
                            print(f"Reached minimum step limit @{dists[i]} (Skipping) SSIM = {d}   ")
                            if skip_ssim_min > d:
                                skip_ssim_min = d
                            skip_count += 1
                        done = i
            # DEBUG
            print("SSIM done!")
            if skip_count > 0:
                print(f"Minimum step limits reached: {skip_count} Worst: {skip_ssim_min}")

        # Save video
        if save_video:
            frames = [np.asarray(images[0])] * lead_inout + [np.asarray(t) for t in images] + [np.asarray(images[-1])] * lead_inout
            clip = ImageSequenceClip.ImageSequenceClip(frames, fps=video_fps)
            filename = f"shift-{shift_number:05}.mp4"
            clip.write_videofile(os.path.join(shift_path, filename), verbose=False, logger=None)

        processed = Processed(p, images if show_images else [], p.seed, initial_info)

        return processed

    def describe(self):
        return "Shift attention in a range of images."
