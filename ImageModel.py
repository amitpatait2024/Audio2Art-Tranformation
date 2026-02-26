%%writefile ImageModel.py
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import librosa as lb
import torch
from typing import Literal
from diffusers import StableDiffusionPipeline
import time

def promptgen(file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Wav2Vec2Tokenizer.from_pretrained('facebook/wav2vec2-base-960h')
    model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h', device_map="auto")


    waveform, rate = lb.load(file, sr=16000)

    input_values = tokenizer(waveform, return_tensors='pt').input_values.to(device)

    logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)

    return transcription[0]
def text2image(
    prompt: str,
    repo_id: Literal[
        "dreamlike-art/dreamlike-photoreal-2.0",
        "hakurei/waifu-diffusion",
        "prompthero/openjourney",
        "stabilityai/stable-diffusion-2-1",
        "runwayml/stable-diffusion-v1-5",
        "nota-ai/bk-sdm-small",
        "CompVis/stable-diffusion-v1-4",
    ],
):
    seed = 2024
    generator = torch.manual_seed(seed)

    NUM_ITERS_TO_RUN = 1
    NUM_INFERENCE_STEPS = 25
    NUM_IMAGES_PER_PROMPT = 1

    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print("Using GPU")
        pipeline = StableDiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(device)
    else:
        print("Using CPU")
        pipeline = StableDiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.float32,
            use_safetensors=True,
        )

    for _ in range(NUM_ITERS_TO_RUN):
        images = pipeline(
            prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
            num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
        ).images
    end = time.time()

    return images[0], start, end
