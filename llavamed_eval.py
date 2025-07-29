# import torch
# from transformers import AutoTokenizer, CLIPImageProcessor
# from llava.model import LlavaMistralForCausalLM
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.utils import disable_torch_init
# from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
# from PIL import Image
# import json
# import os

# def load_model(model_path):
#     """Load LLaVA-Med model"""
#     disable_torch_init()
    
#     model_name = get_model_name_from_path(model_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#     model = LlavaMistralForCausalLM.from_pretrained(
#         model_path, 
#         low_cpu_mem_usage=True, 
#         torch_dtype=torch.float16,
#         device_map="auto"
#     )
#     image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower)
    
#     # Add special tokens for conversation
#     mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
#     mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
#     if mm_use_im_patch_token:
#         tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
#     if mm_use_im_start_end:
#         tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
#     model.resize_token_embeddings(len(tokenizer))
    
#     vision_tower = model.get_vision_tower()
#     if not vision_tower.is_loaded:
#         vision_tower.load_model()
#     vision_tower.to(device='cuda', dtype=torch.float16)
    
#     return model, tokenizer, image_processor

# def generate_response(model, tokenizer, image_processor, image_path, prompt):
#     """Generate response for a single image"""
#     # Load and process image
#     image = Image.open(image_path).convert('RGB')
#     image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    
#     # Prepare conversation
#     conv = conv_templates["llava_v1"].copy()
#     conv.append_message(conv.roles[0], prompt)
#     conv.append_message(conv.roles[1], None)
#     prompt_formatted = conv.get_prompt()
    
#     # Tokenize
#     input_ids = tokenizer_image_token(prompt_formatted, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
#     # Prepare stopping criteria
#     stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#     keywords = [stop_str]
#     stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
#     # Generate
#     with torch.inference_mode():
#         output_ids = model.generate(
#             input_ids,
#             images=image_tensor.unsqueeze(0).half().cuda(),
#             do_sample=True,
#             temperature=0.2,
#             top_p=0.9,
#             max_new_tokens=128,
#             use_cache=True,
#             stopping_criteria=[stopping_criteria]
#         )
    
#     input_token_len = input_ids.shape[1]
#     outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
#     outputs = outputs.strip()
#     if outputs.endswith(stop_str):
#         outputs = outputs[:-len(stop_str)]
#     outputs = outputs.strip()
    
#     return outputs

# # Constants (these should match your LLaVA-Med setup)
# DEFAULT_IMAGE_TOKEN = "<image>"
# DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
# DEFAULT_IM_START_TOKEN = "<im_start>"
# DEFAULT_IM_END_TOKEN = "<im_end>"
# IMAGE_TOKEN_INDEX = -200

# def main():
#     # Model path - update this to your LLaVA-Med model path
#     model_path = "/data/share/models/llava-med-v1.5-mistral-7b"  # 替换为你的LLaVA-Med模型路径
    
#     # Load model
#     model, tokenizer, image_processor = load_model(model_path)
    
#     # Prompt
#     prompt = f"{DEFAULT_IMAGE_TOKEN}\nDescribe the bronchoscopy image in detail, focusing on the appearance and condition of the airway. Include observations such as the presence of new organisms, narrowing, blood clots, sputum, rough surfaces, or normal structures. Note any signs of infiltration changes, congestion, or edema. Mention if the airways appear widened or show signs of external pressure. Describe any surgical stumps, nodules, fistulas, postoperative changes, granulation tissue, necrosis, or masses. Include details of pigmentation, Y-type bifurcations, or ulcers, if present. Provide a comprehensive description of any abnormalities or normal findings observed in the image. Here are some examples:\nA small amount of white mucus sputum can be seen in the lumen\nA cauliflower-like neoplasm was found\nSee a metal coated bronchial support\nThe mucosa can be seen in yellow-white membranes, and the mucosa is slightly congested and edema\nDark red blood marks can be seen in the lumen\nRemember, only output the report, no other words needed."
    
#     # Load test data
#     with open("/data/xingjian_luo/project/zhongshanyi-dataset/huxi_test_qwen_form.json", "r") as f:
#         data = json.load(f)
    
#     eval_result = []
    
#     for i, item in enumerate(data):
#         print(f"Processing item {i+1}/{len(data)}")
        
#         temp_dict = {}
#         image_path = item["images"][0]
        
#         # Generate response
#         output_text = generate_response(model, tokenizer, image_processor, image_path, prompt)
#         print(f"Generated: {output_text}")
        
#         # Store results
#         gt = item["messages"][1]["content"]
#         temp_dict["pred"] = output_text
#         temp_dict["gt"] = gt
#         temp_dict["image_path"] = image_path
#         eval_result.append(temp_dict)
        
#         # Save intermediate results
#     with open("/data/xingjian_luo/project/zhongshanyi-dataset/llavamed_eval_result.json", "w") as f:
#         json.dump(eval_result, f, indent=2, ensure_ascii=False)
            

#     print(f"Evaluation completed. Results saved to llavamed_eval_result.json")

# if __name__ == "__main__":
#     main()


import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
import math
from transformers import set_seed, logging

logging.set_verbosity_error()


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    set_seed(0)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    with open("/data/xingjian_luo/project/zhongshanyi-dataset/huxi_test_qwen_form.json", "r") as f:
        dataset = json.load(f)
    eval_result = []
    for data in tqdm(dataset):
        temp_dict = {}
        image_file = data["images"][0]
        qs ="Describe this image."
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(image_file)
        image_tensor = process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        gt = data["messages"][1]["content"]
        temp_dict["pred"] = outputs
        temp_dict["gt"] = gt
        temp_dict["image_path"] = image_file
        eval_result.append(temp_dict)

    with open("/data/xingjian_luo/project/zhongshanyi-dataset/llavamed_eval_result.json", "w") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)


