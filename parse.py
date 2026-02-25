import re
import json

# Read the file as raw text
with open("manipulate/bignet/legit_discretized.json", "r") as f:
    content = f.read()

# Replace np.float64(15.0) → 15.0
content = re.sub(r'np\.float64\((.*?)\)', r'\1', content)

# Convert single quotes to double quotes (if needed)
content = content.replace("'", '"')

# Load as proper JSON
data = json.loads(content)

# Save cleaned JSON
with open("legit_discretized.json", "w") as f:
    json.dump(data, f)

print("Conversion complete. Saved as legit_clean.json")