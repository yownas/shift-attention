# Shift Attention script for AUTOMATIC1111/stable-diffusion-webui
#
# https://github.com/yownas/shift-attention
#
# Give a prompt like: "photo of (cat:1~0) or (dog:0~1)"
# Generates a sequence of images, lowering the weight of "cat" from 1 to 0 and increasing the weight of "dog" from 0 to 1.
# Will also support multiple numbers. "(cat:1~0~1)" will go from cat:1 to cat:0 to cat:1 streched over the number of steps

import gradio as gr
import imageio
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import re
import sys
import torch
from torchmetrics import StructuralSimilarityIndexMeasure 
from torchvision import transforms
from torch.nn import functional as F
import modules.scripts as scripts
from modules.processing import Processed, process_images, fix_seed
from modules.shared import opts, cmd_opts, state, sd_upscalers
from modules.images import resize_image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rife.RIFE_HDv3 import Model

__ = lambda key, value=None: opts.data.get(f'customscript/shift-attention.py/txt2img/{key}/value', value)

DEFAULT_UPSCALE_METH   = __('Upscaler', 'Lanczos')
DEFAULT_UPSCALE_RATIO  = __('Upscale ratio', 1.0)
CHOICES_UPSCALER  = [x.name for x in sd_upscalers]

class Script(scripts.Script):
    def title(self):
        return "Shift attention"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        steps = gr.Number(label='Steps', value=10)
        with gr.Row():
            video_fps = gr.Number(label='FPS', value=30)
            lead_inout = gr.Number(label='Lead in/out', value=0)
        with gr.Row():
            ssim_diff = gr.Slider(label='SSIM threshold', value=0.0, minimum=0.0, maximum=1.0, step=0.01)
            ssim_ccrop = gr.Slider(label='SSIM CenterCrop%', value=0, minimum=0, maximum=100, step=1)
        with gr.Row():
            rife_passes = gr.Number(label='RIFE passes', value=0)
            rife_drop = gr.Checkbox(label='Drop original frames', value=False)
        with gr.Accordion(label='Shift Attention Extras...', open=False):
            gr.HTML(value='Shift Attention links: <a href=http://github.com/yownas/shift-attention/>Github</a>')
            with gr.Row():
                upscale_meth  = gr.Dropdown(label='Upscaler',    value=lambda: DEFAULT_UPSCALE_METH, choices=CHOICES_UPSCALER)
                upscale_ratio = gr.Slider(label='Upscale ratio', value=lambda: DEFAULT_UPSCALE_RATIO, minimum=0.0, maximum=8.0, step=0.1)
            with gr.Row():
                show_images = gr.Checkbox(label='Show generated images in ui', value=True)
                mirror_mode = gr.Checkbox(label='Mirror mode', value=False)
            substep_min = gr.Number(label='SSIM minimum step', value=0.0001)
            ssim_diff_min = gr.Slider(label='SSIM min threshold', value=75, minimum=0, maximum=100, step=1)
            save_stats = gr.Checkbox(label='Save extra status information', value=True)
            rm_zero_strength = gr.Checkbox(label='Remove Zero strength tags (causes output instability)', value=False)

        return [steps, video_fps, show_images, lead_inout, upscale_meth, upscale_ratio, ssim_diff, ssim_ccrop, ssim_diff_min, substep_min, rife_passes, rife_drop, save_stats, mirror_mode, rm_zero_strength]

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

    def run(self, p, steps, video_fps, show_images, lead_inout, upscale_meth, upscale_ratio, ssim_diff, ssim_ccrop, ssim_diff_min, substep_min, rife_passes, rife_drop, save_stats, mirror_mode, rm_zero_strength):
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
        gen_data = []
        imgcnt=-1
        lead_inout = int(lead_inout)
        if upscale_meth != 'None' and upscale_ratio != 1.0 and upscale_ratio != 0.0:
            tgt_w, tgt_h = round(p.width * upscale_ratio), round(p.height * upscale_ratio)
        else:
            tgt_w, tgt_h = p.width, p.height
        video_fps = 0 if video_fps == None else video_fps
        save_video = video_fps != 0
        ssim_stats = {}
        ssim_stats_new = {}

        if not save_video and not show_images:
            print(f"Nothing to do. You should save the results as a video or show the generated images.")
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

        initial_prompt = p.prompt
        initial_negative_prompt = p.negative_prompt
        initial_seed = p.seed
        cfg_scale = p.cfg_scale

        # Kludge for seed travel 
        p.subseed = p.seed

        # Split prompt and generate list of prompts
        promptlist = re.split("(THEN\([^\)]*\)|THEN)", p.prompt)+[None]
        negative_promptlist = re.split("(THEN\([^\)]*\)|THEN)", p.negative_prompt)+[None]

        # Build new list
        prompts = []
        while len(promptlist) or len(negative_promptlist):
            prompt, subseed, negprompt, negsubseed, new_cfg_scale = (None, None, None, None, None)

            if len(negative_promptlist):
                negprompt = negative_promptlist.pop(0).strip()
                opts = negative_promptlist.pop(0)

                if opts:
                    opts = re.sub('THEN\((.*)\)', '\\1', opts)
                    opts = None if opts == 'THEN' else opts
                    if opts:
                        for then_data in opts.split(','): # Get values from THEN()
                            if '=' in then_data:
                                opt, val = then_data.split('=')
                                if opt == 'seed':
                                    try:
                                        negsubseed = int(val)
                                    except:
                                        negsubseed = None
                                if opt == 'cfg':
                                    try:
                                        new_cfg_scale = float(val)
                                    except:
                                        new_cfg_scale = None

            if len(promptlist):
                prompt = promptlist.pop(0).strip() # Prompt
                opts = promptlist.pop(0) # THEN()
                if opts:
                    opts = re.sub('THEN\((.*)\)', '\\1', opts)
                    opts = None if opts == 'THEN' else opts
                    if opts:
                        for then_data in opts.split(','): # Get values from THEN()
                            if '=' in then_data:
                                opt, val = then_data.split('=')
                                if opt == 'seed':
                                    try:
                                        subseed = int(val)
                                    except:
                                        subseed = None
                                if opt == 'cfg':
                                    try:
                                        new_cfg_scale = float(val)
                                    except:
                                        new_cfg_scale = None

            if not subseed:
                subseed = negsubseed
            prompts += [(prompt, negprompt, subseed, new_cfg_scale)]

        # Set generation helpers
        total_images = int(steps) * len(prompts)
        state.job_count = total_images
        print(f"Generating {total_images} images.")

        # Generate prompt_images and add to images (the big list)
        prompt = p.prompt
        negprompt = p.negative_prompt
        seed = p.seed
        subseed = p.subseed
        cfg_scale = p.cfg_scale
        for new_prompt, new_negprompt, new_subseed, new_cfg_scale in prompts:
            if new_prompt: 
                prompt = new_prompt
            if new_negprompt:
                negprompt = new_negprompt
            if new_subseed:
                subseed = new_subseed

            p.seed = seed
            p.subseed = subseed

            # Frames for the current prompt pair
            prompt_images = []
            dists = []

            # Empty prompt
            if not new_prompt and not new_negprompt: 
                #print("NO PROMPT")
                break

            #DEBUG
            print(f"Shifting prompt:\n+ {prompt}\n- {negprompt}\nSeeds: {int(seed)}/{int(subseed)} CFG: {cfg_scale}~{new_cfg_scale}")
            regex_zero_strength = re.compile("(\([a-z,A-Z_\s\d\-]*:0(\.0)?\)\s?,?)",re.X)

            # Generate the steps
            for i in range(int(steps) + 1):
                if state.interrupted:
                    break
    
                distance = float(i / int(steps))
                p.prompt = shift_attention(prompt, distance)
                p.negative_prompt = shift_attention(negprompt, distance)
                p.subseed_strength = distance
                if not new_cfg_scale is None:
                    p.cfg_scale = cfg_scale * (1.-distance) + new_cfg_scale * distance

                # remove tag groups with zero strength
                if rm_zero_strength:
                    p.prompt = re.sub(regex_zero_strength,"",p.prompt)
                    p.negative_prompt = re.sub(regex_zero_strength,"",p.negative_prompt)

                proc = process_images(p)
                imgcnt += 1
                if initial_info is None:
                    initial_info = proc.info
    
                # upscale - copied from https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel
                if upscale_meth != 'None' and upscale_ratio != 1.0 and upscale_ratio != 0.0:
                    image = [resize_image(0, proc.images[0], tgt_w, tgt_h, upscaler_name=upscale_meth)]
                else:
                    image = [proc.images[0]]
    
                prompt_images += image
                dists += [distance]
                gen_data += [(imgcnt, p.prompt, p.negative_prompt, p.seed, p.subseed, p.subseed_strength, p.cfg_scale)]

            # SSIM
            if ssim_diff > 0:
                ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
                if ssim_ccrop == 0:
                    transform = transforms.Compose([transforms.ToTensor()])
                else:
                    transform = transforms.Compose([transforms.CenterCrop((tgt_h*(ssim_ccrop/100), tgt_w*(ssim_ccrop/100))), transforms.ToTensor()])
    
                transform = transforms.Compose([transforms.ToTensor()])
    
                check = True
                skip_count = 0
                not_better = 0
                skip_ssim_min = 1.0
                min_step = 1.0
    
                done = 0
                while(check):
                    if state.interrupted:
                        break
                    check = False
                    for i in range(done, len(prompt_images)-1):
                        # Check distance between i and i+1
    
                        a = transform(prompt_images[i].convert('RGB')).unsqueeze(0)
                        b = transform(prompt_images[i+1].convert('RGB')).unsqueeze(0)
                        d = ssim(a, b)
    
                        if d < ssim_diff and (dists[i+1] - dists[i]) > substep_min:
                            print(f"SSIM: {dists[i]} <-> {dists[i+1]} = ({dists[i+1] - dists[i]}) {d}")
    
                            # Add image and run check again
                            check = True
    
                            new_dist = (dists[i] + dists[i+1])/2.0
    
                            p.prompt = shift_attention(prompt, new_dist)
                            p.negative_prompt = shift_attention(negprompt, new_dist)
                            p.subseed_strength = new_dist
                            if not new_cfg_scale is None:
                                p.cfg_scale = cfg_scale * (1.-new_dist) + new_cfg_scale * new_dist
    
                            # SSIM stats for the new image
                            ssim_stats_new[(dists[i], dists[i+1])] = d
    
                            print(f"Process: {new_dist}")
                            proc = process_images(p)
                            imgcnt += 1
    
                            if initial_info is None:
                                initial_info = proc.info
                            
                            # upscale - copied from https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel
                            if upscale_meth != 'None' and upscale_ratio != 1.0 and upscale_ratio != 0.0:
                                image = resize_image(0, proc.images[0], tgt_w, tgt_h, upscaler_name=upscale_meth)
                            else:
                                image = proc.images[0]
    
                            # Check if this was an improvment
                            c = transform(image.convert('RGB')).unsqueeze(0)
                            d2 = ssim(a, c)
    
                            if d2 > d or d2 < ssim_diff*ssim_diff_min/100.0:
                                # Keep image if it is improvment or hasn't reached desired min ssim_diff
                                prompt_images.insert(i+1, image)
                                dists.insert(i+1, new_dist)
                                gen_data.insert(i+1, (imgcnt, p.prompt, p.negative_prompt, p.seed, p.subseed, p.subseed_strength, p.cfg_scale))
                            else:
                                print(f"Did not find improvment: {d2} < {d} ({d-d2}) Taking shortcut.")
                                not_better += 1
                                done = i + 1
    
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
                            ssim_stats[(dists[i], dists[i+1])] = d
                # DEBUG
                print("SSIM done!")

                if skip_count > 0:
                    print(f"Minimum step limits reached: {skip_count} Worst: {skip_ssim_min} No improvment: {not_better}")

            # We should have reached the subseed if we were seed traveling
            seed = subseed
            # ..and the cfg
            if new_cfg_scale:
                cfg_scale = new_cfg_scale

            # End of prompt_image loop
            images += prompt_images

        if mirror_mode:
            images = images + images[::-1]

        # Save video before continuing with SSIM-stats and RIFE (If things crashes we will atleast have this video)
        if save_video:
            if mirror_mode:
                images = images + images[::-1]
            try:
                frames = [np.asarray(images[0])] * lead_inout + [np.asarray(t) for t in images] + [np.asarray(images[-1])] * lead_inout
                fps = video_fps if video_fps > 0 else len(frames) / abs(video_fps)
                filename = f"shift-{shift_number:05}.mp4"
                writer = imageio.get_writer(os.path.join(shift_path, filename), fps=fps)
                for frame in frames:
                    writer.append_data(frame)
                writer.close()
            except Exception as err:
                print(f"ERROR: Failed generating video: {err}")

        # SSIM-stats
        if save_stats and ssim_diff > 0:
            # Create scatter plot
            x = []
            y = []
            for i in ssim_stats_new:
                s = i[1] - i[0]
                if s > 0:
                    x.append(s) # step distance
                    y.append(ssim_stats_new[i]) # ssim
            plt.scatter(x, y, s=1, color='#ffa600')
            x = []
            y = []
            for i in ssim_stats:
                s = i[1] - i[0]
                if s > 0:
                    x.append(s) # step distance
                    y.append(ssim_stats[i]) # ssim
            plt.scatter(x, y, s=1, color='#003f5c')
            plt.axvline(substep_min)
            plt.axhline(ssim_diff)

            plt.xscale('log')
            plt.title('SSIM scatter plot')
            plt.xlabel('Step distance')
            plt.ylabel('SSIM')
            filename = f"ssim_scatter-{shift_number:05}.svg"
            plt.savefig(os.path.join(shift_path, filename))
            plt.close()

        # Save settings and other information
        if save_stats:
            D = []

            # Settings
            D.extend(['Prompt:\n', initial_prompt, '\n'])
            D.extend(['Negative prompt:\n', initial_negative_prompt, '\n'])
            D.append('\n')
            D.extend(['Width: ', str(p.width), '\n'])
            D.extend(['Height: ', str(p.height), '\n'])
            D.extend(['Sampler: ', p.sampler_name, '\n'])
            D.extend(['Steps: ', str(p.steps), '\n'])
            D.extend(['CFG scale: ', str(p.cfg_scale), '\n'])
            D.extend(['Seed: ', str(initial_seed), '\n'])
            D.append('- Shift Attention settings ------------\n')
            # Shift Attention Settings
            D.extend(['Steps: ', str(int(steps)), '\n'])
            D.extend(['FPS: ', str(video_fps), '\n'])
            D.extend(['Lead in/out: ', str(int(lead_inout)), '\n'])
            D.extend(['SSIM threshold: ', str(ssim_diff), '\n'])
            D.extend(['SSIM CenterCrop%: ', str(ssim_ccrop), '\n'])
            D.extend(['RIFE passes: ', str(int(rife_passes)), '\n'])
            D.extend(['Drop original frames: ', str(rife_drop), '\n'])
            D.extend(['Upscaler: ', upscale_meth, '\n'])
            D.extend(['Upscale ratio: ', str(upscale_ratio), '\n'])
            D.extend(['SSIM min substep: ', str(substep_min), '\n'])
            D.extend(['SSIM min threshold: ', str(ssim_diff_min), '\n'])
            D.append('---------------------------------------\n')
            # Generation stats
            if ssim_diff:
                D.append(f"Stats: Skip count: {skip_count} Worst: {skip_ssim_min} No improvment: {not_better} Min. step: {min_step}\n")
            D.append(f"Frames: {len(images)}\n")

            # Generation log
            D.append('\n- Generation log ----------------------\n')
            i = 0
            for c,pr,negp,s,ss,d,cfg in gen_data:
                # count, promp, neg_prompt, seed, subseed, strength
                if s == ss:
                    D.append(f"\n--- Frame: {i:05} Image: {c:05} Seed: {s} CFG: {cfg}\n")
                else:
                    D.append(f"\n--- Frame: {i:05} Image: {c:05} Seed: {s} ({ss} {d}) CFG: {cfg}\n")
                D.append(f"+ {pr}\n")
                D.append(f"- {negp}\n")
                i+=1

            filename = f"shift-attention-info-{shift_number:05}.txt"
            file = open(os.path.join(shift_path, filename), 'w')
            file.writelines(D)
            file.close()

        # RIFE (from https://github.com/vladmandic/rife)
        if rife_passes:
            rifemodel = None
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            count = 0

            w, h = tgt_w, tgt_h
            scale = 1.0
            fp16 = False
    
            tmp = max(128, int(128 / scale))
            ph = ((h - 1) // tmp + 1) * tmp
            pw = ((w - 1) // tmp + 1) * tmp
            padding = (0, pw - w, 0, ph - h)
    
            def rifeload(model_path: str = os.path.dirname(os.path.abspath(__file__)) + '/rife/flownet-v46.pkl', fp16: bool = False):
                global rifemodel # pylint: disable=global-statement
                torch.set_grad_enabled(False)
                if torch.cuda.is_available():
                    torch.backends.cudnn.enabled = True
                    torch.backends.cudnn.benchmark = True
                    if fp16:
                        torch.set_default_tensor_type(torch.cuda.HalfTensor)
                rifemodel = Model()
                rifemodel.load_model(model_path, -1)
                rifemodel.eval()
                rifemodel.device()
    
            def execute(I0, I1, n):
                global rifemodel # pylint: disable=global-statement
                if rifemodel.version >= 3.9:
                    res = []
                    for i in range(n):
                        res.append(rifemodel.inference(I0, I1, (i+1) * 1. / (n+1), scale))
                    return res
                else:
                    middle = rifemodel.inference(I0, I1, scale)
                    if n == 1:
                        return [middle]
                    first_half = execute(I0, middle, n=n//2)
                    second_half = execute(middle, I1, n=n//2)
                    if n % 2:
                        return [*first_half, middle, *second_half]
                    else:
                        return [*first_half, *second_half]
    
            def pad(img):
                return F.pad(img, padding).half() if fp16 else F.pad(img, padding)
    
            rife_images = frames
    
            for i in range(int(rife_passes)):
                print(f"RIFE pass {i+1}")
                if rifemodel is None:
                    rifeload()
                print('Interpolating', len(rife_images), 'images')
                frame = rife_images[0]
                buffer = []
    
                I1 = pad(torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
                for frame in rife_images:
                    I0 = I1
                    I1 = pad(torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
                    output = execute(I0, I1, 1)
                    for mid in output:
                        mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                        buffer.append(np.asarray(mid[:h, :w]))
                    if not rife_drop:
                        buffer.append(np.asarray(frame))
                rife_images = buffer
    
            if mirror_mode:
                rife_images = rife_images + rife_images[::-1]

            try:
                frames = [np.asarray(rife_images[0])] * lead_inout + [np.asarray(t) for t in rife_images] + [np.asarray(rife_images[-1])] * lead_inout
                fps = video_fps if video_fps > 0 else len(frames) / abs(video_fps)
                filename = f"shift-rife-{shift_number:05}.mp4"
                writer = imageio.get_writer(os.path.join(shift_path, filename), fps=fps)
                for frame in frames:
                    writer.append_data(frame)
                writer.close()
            except Exception as err:
                print(f"ERROR: Failed generating RIFE video: {err}")
        # RIFE end

        processed = Processed(p, images if show_images else [], p.seed, initial_info)

        return processed

    def describe(self):
        return "Shift attention in a range of images."
