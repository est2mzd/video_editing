import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import numpy as np

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# LoRA読み込み
pipe.load_lora_weights(
    "/workspace/weights/lora",
    weight_name="anime.safetensors"
)

# pipe.set_adapters(["default"], adapter_weights=[0.8])
pipe.set_adapters(["default_0"], adapter_weights=[0.8])


def stylize(frame, prompt):
    image = Image.fromarray(frame).resize((512, 512))

    result = pipe(
        prompt=prompt,
        image=image,
        strength=0.5,
        guidance_scale=7.5
    ).images[0]

    return np.array(result)


if __name__ == "__main__":
    # /workspace/src/test/girl_resized.png を読み込んで複数スタイルを出力する
    input_image_path = "/workspace/src/test/girl_resized.png"

    styles = [
        "ukiyo-e",
        "ghibli",
        "pixel_art",
        "anime",
        "cyberpunk",
        "watercolor",
        "oil_painting",
        "american_comic",
    ]

    style_prompts = {
        style: f"apply style of {style.replace('_', ' ')}"
        for style in styles
    }

    frame = np.array(Image.open(input_image_path))

    for style in styles:
        prompt = style_prompts[style]
        output_image_path = (
            f"/workspace/src/test/girl_stylized_lora_{style}.png"
        )
        stylized_frame = stylize(frame, prompt)
        Image.fromarray(stylized_frame).save(output_image_path)
        print(f"[{style}] prompt: {prompt}")
        print(f"Stylized image saved to {output_image_path}")
