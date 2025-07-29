from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
import json


# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/data/xingjian_luo/project/LLaMA-Factory/saves/qwen2.5_vl-7b/full/sft_v3", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processor
processor = AutoProcessor.from_pretrained("/data/xingjian_luo/project/LLaMA-Factory/saves/qwen2.5_vl-7b/full/sft_v3")


prompt = """Describe the bronchoscopy image in detail, focusing on the appearance and condition of the airway. 
Include observations such as the presence of new organisms, narrowing, blood clots, sputum, rough surfaces, or normal structures. Note any signs of infiltration changes, congestion, or edema. Mention if the airways appear widened or show signs of external pressure. Describe any surgical stumps, nodules, fistulas, postoperative changes, granulation tissue, necrosis, or masses. Include details of pigmentation, Y-type bifurcations, or ulcers, if present. 
Provide a comprehensive description of any abnormalities or normal findings observed in the image. Here are some examples:
A small amount of white mucus sputum can be seen in the lumen
A cauliflower-like neoplasm was found
See a metal coated bronchial support
The mucosa can be seen in yellow-white membranes, and the mucosa is slightly congested and edema
Dark red blood marks can be seen in the lumen
Remember, only output the report, no other words needed."""

with open("/data/xingjian_luo/project/zhongshanyi-dataset/huxi_test_qwen_form.json", "r") as f:
    data = json.load(f)

eval_result = []
for i in data:
    temp_dict = {}
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": i["images"][0],
                    "min_pixels": 224 * 224,
                    "max_pixels": 1280 * 28 * 28,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

    gt = i["messages"][1]["content"]
    temp_dict["pred"] = output_text[0]
    temp_dict["gt"] = gt
    temp_dict["image_path"] = i["images"][0]
    eval_result.append(temp_dict)
    with open("/data/xingjian_luo/project/zhongshanyi-dataset/qwen_ft_eval_result.json", "w") as f:
        f.write(json.dumps(eval_result, indent=2))
    






