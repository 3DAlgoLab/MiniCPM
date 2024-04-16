import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("openbmb/MiniCPM-V-2", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V-2", trust_remote_code=True)
model.eval().cuda()

image = Image.open("./robot.png").convert("RGB")
question = "What is in the image?"
msgs = [{"role": "user", "content": question}]

res, context, _ = model.chat(
    image=image,
    msgs=msgs,
    context=None,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7,
)
print(res)
