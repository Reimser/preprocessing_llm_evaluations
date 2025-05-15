import json
from preprocessing.variants import to_freitext, to_bulletpoints, to_json_format
from models.schema import ZeugnisInfo

# Load example
with open("data/input_profiles.json", "r") as f:
    profiles = json.load(f)

# Test one profile
profile = ZeugnisInfo(**profiles[0])

print("Freitext:\n", to_freitext(profile))
print("\nBulletpoints:\n", to_bulletpoints(profile))
print("\nJSON:\n", to_json_format(profile))
