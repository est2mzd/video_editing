import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import numpy as np

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

def stylize(frame, prompt):
    image = Image.fromarray(frame)
    image = image.resize((512, 512))

    result = pipe(
        prompt=prompt,
        image=image,
        strength=0.6,
        guidance_scale=7.5
    ).images[0]

    return np.array(result)


if __name__ == "__main__":
    # /workspace/src/test/girl_resized.png を読み込んで処理する
    input_image_path = "/workspace/src/test/girl_resized.png"
    output_image_path = "/workspace/src/test/girl_stylized_diffusion.png"
    prompt = "apply style of oil painting"  #"A beautiful painting of a girl"

    frame = np.array(Image.open(input_image_path))
    stylized_frame = stylize(frame, prompt)
    Image.fromarray(stylized_frame).save(output_image_path)
    print(f"Stylized image saved to {output_image_path}")