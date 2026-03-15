import base64

with open("/Users/iwon-yong/dev/workspace/ai/pytorchtrf/datasets/pet/test/dog/dog.4001.jpg", "rb") as f:
    encoded = base64.b64encode(f.read()).decode("utf-8")

print(encoded)