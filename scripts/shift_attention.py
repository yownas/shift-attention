# Shift Attention script for AUTOMATIC1111/stable-diffusion-webui
#
# https://github.com/yownas/shift-attention
#
# Give a prompt like: "photo of (cat:1~0) or (dog:0~1)"
# Generates a sequence of images, lowering the weight of "cat" from 1 to 0 and increasing the weight of "dog" from 0 to 1.

import os
import re
import modules.scripts as scripts
import gradio as gr
import math
import numpy
import random
from modules.processing import Processed, process_images, fix_seed
from modules.shared import opts, cmd_opts, state


class Script(scripts.Script):
    def title(self):
        return "Shift attention"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        steps = gr.Number(label='Steps', value=10)

        show_images = gr.Checkbox(label='Show generated images in ui', value=True)
        save_video = gr.Checkbox(label='Save results as video', value=True)
        video_fps = gr.Number(label='Frames per second', value=30)

        return [steps, save_video, video_fps, show_images]

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

    def run(self, p, steps, save_video, video_fps, show_images):
        re_attention_span = re.compile(r"([\-.\d]+)~([\-.\d]+)", re.X)

        def shift_attention(text, distance):

            def inject_value(distance, match_obj):
                start_weight = float(match_obj.group(1))
                end_weight = float(match_obj.group(2))
                return str(start_weight + (end_weight - start_weight) * distance)

            res = re.sub(re_attention_span, lambda match_obj: inject_value(distance, match_obj), text)
            return res

        initial_info = None
        images = []

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

        # Custom seed travel saving
        shift_path = os.path.join(p.outpath_samples, "shift")
        os.makedirs(shift_path, exist_ok=True)
        shift_number = Script.get_next_sequence_number(shift_path)
        shift_path = os.path.join(shift_path, f"{shift_number:05}")
        p.outpath_samples = shift_path

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

        for i in range(int(steps) + 1):
            if state.interrupted:
                break

            p.prompt = shift_attention(initial_prompt, float(i / int(steps)))

            proc = process_images(p)
            if initial_info is None:
                initial_info = proc.info
            images += proc.images

        if save_video:
            clip = ImageSequenceClip.ImageSequenceClip([np.asarray(i) for i in images], fps=video_fps)
            clip.write_videofile(os.path.join(shift_path, f"shift-{shift_number:05}.mp4"), verbose=False, logger=None)

        processed = Processed(p, images if show_images else [], p.seed, initial_info)

        return processed

    def describe(self):
        return "Shift attention in a range of images."
