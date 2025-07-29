import json
import os
import shutil
from openai import OpenAI
# import googletrans
# from googletrans import Translator


client = OpenAI(
    api_key="your API Key",
    base_url="model deployment link"
)


prompt_location = "Please use professional anatomical terms to translate the bronchus locations into English. Please only output the English translation, without any other content. Translation location: {location}. Translation content:"
prompt_report = "Please use professional bronchoscopy examination report terms to translate the report into English. Please only output the English translation, without any other content. Translation report: {report}. Translation content:"




with open("datapath","r") as f:
    train_data = [json.loads(line) for line in f]

# 路径设置
source_folder = "source_folder"
destination_folder = "destination_folder"

def translate(location):
    try:
        
        completion = client.chat.completions.create(
            model="model_name",
            messages=[
                {
                    "role": "user",
                    "content": prompt_location.format(location=location)
                }
            ]
        )
        
        processed_data = completion.choices[0].message.content
        return processed_data
        
    except Exception as e:
        print(f"error: {e}")


new_dataset = []

for idx, entry in enumerate(train_data):
    original_image_name = os.path.basename(entry["image"])
    idx = idx + 6099
    new_image_id = f"{idx + 1:06d}"  
    new_image_name = f"{new_image_id}.png"
    new_image_path = os.path.join(destination_folder, new_image_name)
    
    source_image_path = os.path.join(source_folder, original_image_name)
    if os.path.exists(source_image_path):
        shutil.copy(source_image_path, new_image_path)
    else:
        print(f"Warning: Source image {source_image_path} not found.")
        continue
    
    location = entry["image"], "Unknown Location"
    location = location.split("/")[-1].split(".")[0]  
    location_en = translate(location)

    caption = entry["conversations"][-1]["value"]
    
    new_entry = {
        "image_path": new_image_path,
        "image_id": new_image_id,
        "caption": caption,
        "location": location_en,
        "width": entry["width"],
        "height": entry["height"]
    }
    
    new_dataset.append(new_entry)

new_dataset_path = "new_dataset_path"
with open(new_dataset_path, "w", encoding="utf-8") as f:
    json.dump(new_dataset, f, indent=4, ensure_ascii=False)

print(f"New dataset created with {len(new_dataset)} entries at {new_dataset_path}")