import json
from collections import Counter

with open("path of your dataset", "r", encoding="utf-8") as f:
    labeled_dataset = json.load(f)

category_counter = Counter()

for item in labeled_dataset:
    labels = item["label"].split(",")
    labels = [label.strip() for label in labels]
   
    category_counter.update(labels)

sorted_categories = sorted(category_counter.items(), key=lambda x: x[1], reverse=True)

print("Category Statistics (sorted from most to least):")
for category, count in sorted_categories:
    print(f"{category}: {count}")

sorted_statistics = {category: count for category, count in sorted_categories}
with open("sorted_category_statistics.json", "w", encoding="utf-8") as f:
    json.dump(sorted_statistics, f, ensure_ascii=False, indent=2)