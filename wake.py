import os
import click
import youtube_dl
import pygame
from pydub import AudioSegment
from moviepy.editor import AudioFileClip
from threading import Thread
from quran_metadata import quran_chapter_to_verse
import time
import subprocess
import openai
import random
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")
COMPLETIONS_MODEL = "gpt-3.5-turbo"

@click.command()
@click.option('--url', required=True, help='YouTube video URL.')
@click.option('--hours', required=True, help='Countdown hours.')
@click.option('--output', required=True, help='Name of the output file.')

def main(url, hours, output):
    # Download video
    download_video(url,output)

    # test audio
    test_audio(f'{output}.mp3',output)

    # Start countdown
    countdown(hours)

    # Play audio with fade-in effect
    play_audio(f'{output}.mp3')

def download_video(url,output):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{output}.mp3',  # name of the audio file
        'quiet': False,
        'verbose': True,
    }
    if not os.path.exists(f'{output}.mp3'):
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

def test_audio(file_path,output):
    # rename the {output}.mp3 file to temp.mp3
    if not os.path.exists(f'{output}.mp3'):
        os.rename(f'{output}.mp3', 'temp.mp3')
        command = f'ffmpeg -i temp.mp3 {output}.mp3'
        # remove the temp.mp3 file
        subprocess.run(command, shell=True)
        os.remove('temp.mp3')
        try:
            audio = AudioSegment.from_file(file_path, format="mp3")
        except:
            print("Error: Audio file could not be loaded.")

def countdown(hours):
    countdown_seconds = float(hours) * 60 * 60
    # use tqdm to display the countdown progress
    for i in tqdm(range(int(countdown_seconds))):
        time.sleep(1)

def get_verses_and_explanations():
    # Get the verses
    verses = select_quran_verse()
    # print(f"Verse 1:\n{verses}")

    # initialize the verse_texts output and explanations output
    verse_texts = []
    explanations = []

    # Get the explanations
    for verse in verses:
        prompt = f"You are an Islamic Scholar with extensive knowledge on the Quran and the life of Prophet Muhammad (pbuh). Please simply display {verse} in the closest English translation to the best of your knowledge, and nothing else."
        verse_text = query_gpt(prompt)
        prompt2 = f"You are an Islamic Scholar with extensive knowledge on the Quran and the life of Prophet Muhammad (pbuh). Please provide the Tafsir (meaning) of {verse_text}."
        explanation = query_gpt(prompt2)
        verse_texts.append(verse_text)
        explanations.append(explanation)
    
    # begin the interactive session
    for verse_text, explanation, verse in zip(verse_texts, explanations, verses):
        print(f'\n\n\n\n{verse}:\n{verse_text} \n\n ------------------ \n\n When you are ready to see the explanaton, press Enter.')
        # wait for the user to press Enter
        input()

        # if not last verse:
        if verse != verses[-1]:
            print(f'{explanation} \n\n ------------------ \n\n When you are ready to proceed to the next verse, press Enter (20 second timer).')
        else:
            print(f'{explanation} \n\n ------------------ \n\n When you are ready to stop the alarm, press Enter (20 second timer).')
        time.sleep(20) # Wait for the user to press Enter, but not before 20 seconds have passed
        input() # wait for the user to press Enter

def select_quran_verse():
    # List of surah numbers
    surahs = list(quran_chapter_to_verse.keys())

    # Number of verses in each surah, this is just a placeholder
    # Please replace this with the actual number of verses in each surah
    verses_in_surah = list(quran_chapter_to_verse.values())

    # Select a random surah
    surah = random.choice(surahs)

    # Use the index of the surah to find the number of verses in the surah
    num_verses = verses_in_surah[surah - 1]

    # Check if the surah has at least 3 verses
    if num_verses < 3:
        # raise an error saying there are not enough verses in the Surah
        raise Exception(f'There are not enough verses in Surah {surah}.')

    # Select a verse such that there are at least 2 verses that follow after the verse
    verse = random.randint(1, num_verses - 2)

    # Return the verses
    return (f'{surah}:{verse}', f'{surah}:{verse + 1}', f'{surah}:{verse + 2}')

def query_gpt(prompt):
    # GPT-3 API parameters
    COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.4 because it gives the most predictable, factual answer with some variation.
    "temperature": 0.4,
    "max_tokens": 1000,
    "model": COMPLETIONS_MODEL,
    }
    
    response = openai.ChatCompletion.create(
                messages = [{"role": "user", "content": prompt}],
                **COMPLETIONS_API_PARAMS
            )

    return response['choices'][0]['message']['content']

def gradually_decrease_volume():
    # Gradually decrease the volume over 5 seconds
    for i in range(1, 501):  # 500 steps over 5 seconds
        pygame.mixer.music.set_volume(1 - i / 500.0)  # gradually decrease the volume
        time.sleep(0.01)

def stop_audio():
    # Gradually decrease the volume over 5 seconds
    get_verses_and_explanations()

    # Start a new thread to gradually decrease the volume
    Thread(target=gradually_decrease_volume).start()

    # Stop the audio
    pygame.mixer.music.stop()

def play_audio(file_path):
    # Initialize pygame mixer
    pygame.mixer.init()

    # Load the audio file
    pygame.mixer.music.load(file_path)

    # Start playing the audio with volume 0
    pygame.mixer.music.set_volume(0.0)
    pygame.mixer.music.play()

    # Start a new thread to gradually stop the audio
    stop_audio_thread = Thread(target=stop_audio)
    stop_audio_thread.start()

    # Gradually increase the volume over 15 minutes
    for i in range(1, 901):  # 900 seconds = 15 minutes
        time.sleep(1)
        pygame.mixer.music.set_volume(i / 900.0)  # gradually increase the volume
        # break the loop if the stop_audio thread has finished
        if not stop_audio_thread.is_alive():
            break
    
    # if the audio reaches the end, restart from the beginning until the stop_audio thread has finished
    while pygame.mixer.music.get_busy():
        pygame.mixer.music.rewind()
        time.sleep(1)
        # break the loop if the stop_audio thread has finished
        if not stop_audio_thread.is_alive():
            break

    # Wait for the stop_audio thread to finish
    stop_audio_thread.join()
    time.sleep(5)

    # Stop the audio
    pygame.mixer.music.stop()

if __name__ == "__main__":
    main()
