import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from MeloTTS.melo.api import TTS
from MeloTTS.melo.api import TTS
from diffusers import DiffusionPipeline
from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_videoclips

def create_video_from_images(
    image_paths,
    image_durations,
    audio_file,
    output_video_path="output.mp4",
    fps=30,
):
    """
    Create a video from a sequence of images and an audio file.

    Args:
        image_paths (list): A list of file paths to the images.
        image_durations (list): A list of durations (in seconds) for each image.
        audio_file (str): The file path to the audio file.
        transition_duration (float, optional): The duration (in seconds) of the transition between images. Defaults to 1.
        output_video_path (str, optional): The output file path for the video. Defaults to 'output.mp4'.
        fps (int, optional): The frames per second for the video. Defaults to 30.

    Returns:
        None
    """
    # Load the audio file
    audio_clip = AudioFileClip(audio_file)
    frame_duration = audio_clip.duration / len(image_paths)

    # Create an ImageSequenceClip from the image paths
    image_clips = [
        ImageSequenceClip([img_path], durations=[frame_duration])
        for img_path, _ in zip(image_paths, image_durations)
    ]

    # Create a concatenated video clip from the image clips
    video_clip = concatenate_videoclips(image_clips)

    # Set the audio clip to the video clip
    video_clip = video_clip.set_audio(audio_clip)

    # Write the video to the output file
    video_clip.write_videofile(output_video_path, fps=fps)


def generate_image(index, img_prompt):
    model_id = "cagliostrolab/animagine-xl-3.1"
    print("⚙️ Generating image " + str(index))
    pipe = DiffusionPipeline.from_pretrained(model_id, use_safetensors=True).to("mps")

    story_gen_prompt = img_prompt + ". Use comic style for images."
    negative_prompt = "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"

    image = pipe(
        story_gen_prompt,
        negative_prompt=negative_prompt,
        width=720,
        height=1280,
        guidance_scale=7,
        num_inference_steps=28,
    ).images[0]

    img_path = "scene" + str(index) + ".png"
    image.save(img_path)
    
    return img_path

def generate_audio(script: str):
    model = TTS(language='EN_V2', device='cpu')
    speaker_ids = model.hps.data.spk2id

    output_path = 'audio.mp3'
    model.tts_to_file(script, speaker_ids['EN-INDIA'], output_path, speed=1.0)


llm = Ollama(model="mistral")
audio_file = "audio.mp3"
output_path = "output.mp4"
image_durations = [7, 22, 20, 20, 20, 20]

story_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a talented short video scriptwriter known for creating viral content. Your scripts are engaging and optimized for videos lasting around 1 minute, captivating audiences with every second. Whenever asked for a script you provide simple English sentences without details of what scene it is or when it ends and don't break it into list give direct paragraphs.",
        ),
        ("user", "{input}"),
    ]
)

story_fmt_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Your role is to describe the given story as separate detailed scenes in JSON format where keys will be incremental numbers and values will be a string describing the scene as a prompt for that scene's image generation. Each scene should be described in plain, expressive language, serving as a prompt for generating scene images. Be as detailed as possible when describing the scene""",
        ),
        ("user", "{input}"),
    ]
)

story_chain = story_gen_prompt | llm
formatter_chain = story_fmt_prompt | llm

story = story_chain.invoke({"input": "Write a funny short story for kids."})
print("Story generated ✅")
print(story)

formatted_story = formatter_chain.invoke({"input": story})
print("Story formatted 👗")
print(formatted_story)
converted_json = json.loads(formatted_story)

# Image generation
image_prompts = [converted_json[key] for key in converted_json]
image_paths = [generate_image(index, image) for index, image in enumerate(image_prompts)]

# Audio generation
generate_audio(story)

# Video generation [merges video and audio]
create_video_from_images(
    image_paths,
    image_durations=image_durations,
    audio_file=audio_file,
    output_video_path=output_path,
)

# Cleanup logic (delete standalone images, audio)
os.remove(audio_file)
for image in image_paths:
    os.remove(image)
