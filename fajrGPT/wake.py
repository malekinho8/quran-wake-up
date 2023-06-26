import os
import sys; sys.path.append('./')
import click
import youtube_dl
import pygame
from pydub import AudioSegment
from threading import Thread
from fajrGPT.quran_metadata import quran_chapter_to_verse
import time
import subprocess
import openai
import random
from tqdm import tqdm
from Quran_Module import Project_Quran

openai.api_key = os.getenv("OPENAI_API_KEY")
COMPLETIONS_MODEL = "gpt-3.5-turbo"

@click.command()
@click.option('--url', required=True, help='YouTube video URL.')
@click.option('--time', required=True, help='Countdown time in format [number][h/m/s], i.e. 1h would create a 1 hour timer.')
@click.option('--output', required=True, help='Name of the output file.')

def main(url, time, output):
    # Download video
    flag = download_video(url,output)

    # test audio
    test_audio(f'{output}.mp3',output,flag)

    # convert time to seconds
    countdown_seconds = convert_to_seconds(time)

    # Start countdown
    countdown(countdown_seconds)

    # Play audio with fade-in effect on a separate thread
    play_audio_thread = Thread(target=play_audio, args=(f'{output}.mp3',))
    play_audio_thread.start()

    # display the quran verses
    get_verses_and_explanations()

    # stop the audio once the user has completed reading the verses
    stop_audio()

    # return back to the main thread
    play_audio_thread.join()

def convert_to_seconds(time_str):
    # Get the number and the time unit (h/m/s)
    number = float(time_str[:-1])
    unit = time_str[-1].lower()

    # Convert to seconds
    if unit == "h":
        return number * 60 * 60
    elif unit == "m":
        return number * 60
    elif unit == "s":
        return number
    else:
        raise ValueError("Invalid time format. Use [number][h/m/s] format.")


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
        converted_flag = False
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    else:
        converted_flag = True
    return converted_flag

def test_audio(file_path,output,converted_flag):
    # rename the {output}.mp3 file to temp.mp3
    if not converted_flag:
        os.rename(f'{output}.mp3', 'temp.mp3')
        command = f'ffmpeg -i temp.mp3 {output}.mp3'
        # remove the temp.mp3 file
        subprocess.run(command, shell=True)
        os.remove('temp.mp3')
        try:
            audio = AudioSegment.from_file(file_path, format="mp3")
        except:
            print("Error: Audio file could not be loaded.")
    else:
        print(f'{output}.mp3 has already been downloaded and converted.')

def countdown(countdown_seconds):
    # use tqdm to display the countdown progress
    print('\n\n\n\n ---------------- BEGINNING COUNTDOWN ---------------- \n\n\n\n')
    # print the current time in HH:MM format
    print(f'\n\nSTART TIME: {time.strftime("%H:%M", time.localtime())}\n\n')
    for i in tqdm(range(int(countdown_seconds))):
        time.sleep(1)
    print('\n\n\n\n ---------------- COUNTDOWN COMPLETE ----------------')
    # print the current time in HH:MM format
    print(f'\n\nEND TIME: {time.strftime("%H:%M", time.localtime())}\n\n')

def get_verses_and_explanations():
    # Get the verses
    verses_Quran_Module, verses  = select_quran_verse()
    # print(f"Verse 1:\n{verses}")

    # initialize the verse_texts output and explanations output
    verse_texts = []
    explanations = []

    # Get the explanations
    for verse_QM, verse in zip(verses_Quran_Module, verses):
        verse_text = Project_Quran().Get_Ayah_English(verse_QM).split('"')[1][0:-1]
        prompt2 = f"To the best of your knowledge, please provide the Tafsir (meaning) of {verse_text} which comes from verse {verse} of the Quran. If you know of any other alternative translations of the verse in question, please provide that too, but if you are not familiar with this verse, simply try to explain the language of the verse."
        explanation = query_gpt(prompt2)
        verse_texts.append(verse_text)
        explanations.append(explanation)
    
    # begin the interactive session
    for verse_text, explanation, verse in zip(verse_texts, explanations, verses):
        print(f'\n\n\n\n\n\n ------------------ FIRST VERSE ------------------') if verse == verses[0] else None
        print(f'\n\n{verse}:\n{verse_text} \n\n ------------------ \n\n When you are ready to see the explanaton, press Enter.')
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
    return (f'59,{surah},{verse}', f'59,{surah},{verse + 1}', f'59,{surah},{verse + 2}'), (f'{surah}:{verse}',f'{surah}:{verse+1}' ,f'{surah}:{verse+2}') # 59 corresponds to english Quran verse

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

def gradually_change_volume(start_volume, end_volume, duration):
    # Compute the number of steps and the change in volume per step
    steps = duration
    delta_volume = (end_volume - start_volume) / steps

    # Set the initial volume
    pygame.mixer.music.set_volume(start_volume)

    # Gradually change the volume
    for i in range(steps):
        # Wait for 1 second
        time.sleep(1)

        # Change the volume
        new_volume = start_volume + i * delta_volume
        pygame.mixer.music.set_volume(new_volume)

        # If the stop_audio function has been called, break the loop
        if stop_audio_called:
            print("Stopping volume change due to stop_audio being called")
            break

def play_audio(file_path):
    global stop_audio_called
    stop_audio_called = False

    # Initialize pygame mixer
    pygame.mixer.init()

    # Load the audio file
    pygame.mixer.music.load(file_path)

    # Start playing the audio with volume 0
    pygame.mixer.music.set_volume(0.0)
    pygame.mixer.music.play()

    # Gradually increase the volume over 15 minutes in a separate thread
    gradually_change_volume(0.0, 1.0, 900)

    # Loop the audio until the stop_audio function is called
    while not stop_audio_called:
        if not pygame.mixer.music.get_busy():
            # The music has finished, restart it
            pygame.mixer.music.play()
        time.sleep(1)    

def stop_audio():
    global stop_audio_called
    stop_audio_called = True # stop the audio in the other thread
    time.sleep(1)
    stop_audio_called = False # reset back to False to allow volume to decrease
    gradually_change_volume(pygame.mixer.music.get_volume(), 0.0, 10)

    # Stop the audio completely
    stop_audio_called = True
    pygame.mixer.music.stop()

if __name__ == "__main__":
    main()

