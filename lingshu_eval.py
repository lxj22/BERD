from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import PIL
from transformers import AutoProcessor
import json



processor = AutoProcessor.from_pretrained("/data2/xingjian_luo/checkpoint/Lingshu-7B")
llm = LLM(model="/data2/xingjian_luo/checkpoint/Lingshu-7B", limit_mm_per_prompt = {"image": 4}, tensor_parallel_size=2, enforce_eager=True, trust_remote_code=True,)
sampling_params = SamplingParams(
            temperature=0.7,
            top_p=1,
            repetition_penalty=1,
            max_tokens=1024,
            stop_token_ids=[],
        )


with open("/data/xingjian_luo/project/zhongshanyi-dataset/huxi_test_qwen_form.json","r") as f:
    data = json.load(f)

text = """Describe the bronchoscopy image in detail, focusing on the appearance and condition of the airway. 
Include observations such as the presence of new organisms, narrowing, blood clots, sputum, rough surfaces, or normal structures. Note any signs of infiltration changes, congestion, or edema. Mention if the airways appear widened or show signs of external pressure. Describe any surgical stumps, nodules, fistulas, postoperative changes, granulation tissue, necrosis, or masses. Include details of pigmentation, Y-type bifurcations, or ulcers, if present. 
Provide a comprehensive description of any abnormalities or normal findings observed in the image. Here are some examples:
A small amount of white mucus sputum can be seen in the lumen
A cauliflower-like neoplasm was found
See a metal coated bronchial support
The mucosa can be seen in yellow-white membranes, and the mucosa is slightly congested and edema
Dark red blood marks can be seen in the lumen
Remember, only output the report, no other words needed."""
eval_result = []
for i in data:
    temp_dict = {}
    image_path = i["images"][0]
    image = PIL.Image.open(image_path)

    message = [
        {
            "role":"user",
            "content":[
                {"type":"image","image":image},
                {"type":"text","text":text}
                ]
                }
    ]
    prompt = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(message)
    mm_data = {}
    mm_data["image"] = image_inputs
    processed_input = {
    "prompt": prompt,
    "multi_modal_data": mm_data,
    }
    gt = i["messages"][1]["content"]
    outputs = llm.generate([processed_input], sampling_params=sampling_params)
    temp_dict["pred"] = outputs[0].outputs[0].text
    temp_dict["gt"] = gt
    temp_dict["image_path"] = image_path
    print(outputs[0].outputs[0].text)
    eval_result.append(temp_dict)
    with open("lingshu_eval_result.json", "w") as f:
        f.write(json.dumps(eval_result,indent=2))
        
