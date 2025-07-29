import json
import os
from openai import OpenAI
from tqdm import tqdm
client = OpenAI(
    api_key="Your API key",
    base_url="model deployment link"
)

prompt = """You are a doctor skilled in writing bronchoscopic examination reports. You need to classify the following report content. The text may include at least one or more categories. You only need to output the category names. If there are multiple category labels, separate them with a comma, such as "new organism","blood". Below are the categories:
new organism,
narrow,
blood,
clot,
sputum,
rough,
normal,
infiltration changes,
congested,
edematous,
widened,
external pressure,
surgical stump,
nodules,
fistula,
Postoperative change,
granulation,
necrotic,
mass,
pigmentation,
tube,
ulcer.
Here are the sentences that need classification:{sentence}"""

def annotate(sentence):
    try:
        
        completion = client.chat.completions.create(
            model="your model",
            messages=[
                {
                    "role": "user",
                    "content": prompt.format(sentence=sentence)
                }
            ]
        )
        
        processed_data = completion.choices[0].message.content
        return processed_data
        
    except Exception as e:
        print(f"error: {e}")


with open("dataset path","r",encoding="utf-8") as f:
    dataset = json.load(f)


for i in tqdm(dataset):
    i["label"] = annotate(i["caption"])
    with open("dataset output","w") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


