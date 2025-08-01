import os
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

prompt = """Describe the bronchoscopy image in detail, focusing on the appearance and condition of the airway. 
Include observations such as the presence of new organisms, narrowing, blood clots, sputum, rough surfaces, or normal structures. Note any signs of infiltration changes, congestion, or edema. Mention if the airways appear widened or show signs of external pressure. Describe any surgical stumps, nodules, fistulas, postoperative changes, granulation tissue, necrosis, or masses. Include details of pigmentation, Y-type bifurcations, or ulcers, if present. 
Provide a comprehensive description of any abnormalities or normal findings observed in the image. Here are some examples:
A small amount of white mucus sputum can be seen in the lumen
A cauliflower-like neoplasm was found
See a metal coated bronchial support
The mucosa can be seen in yellow-white membranes, and the mucosa is slightly congested and edema
Dark red blood marks can be seen in the lumen
Remember, only output the report, no other words needed."""


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

path = "/data2/xingjian_luo/checkpoint/InternVL3-8B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


with open("/data/xingjian_luo/project/zhongshanyi-dataset/huxi_test_qwen_form.json") as f:
    dataset_test = json.load(f)


answer_list = []
for i in tqdm(dataset_test):
    image_path = i["images"][0]
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    # single-image single-round conversation (单图单轮对话)
    question = '<image>\n'+prompt
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    temp  =  {}
    temp["pred"] = response
    temp["gt"] = i["messages"][1]["content"]
    temp["image_path"] = image_path
    # print(f'User: {question}\nAssistant: {response}')
    # print("ground truth: ",i["messages"][1]["content"])
    answer_list.append(temp)
    

    with open("internvl_eval_result.json","w",encoding="utf-8") as f:
        f.write(json.dumps(answer_list,indent=2,ensure_ascii=False))