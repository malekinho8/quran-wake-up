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
@click.option('--names-flag', required=False, help='Whether or not to include a randomly selected name of Allah in the preamble.', default=True)

def main(url, time, output, names_flag):
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

    # display a name of Allah
    get_name_of_allah_and_explanation(names_flag)

    # display the quran verses
    get_verses_and_explanations(countdown_seconds)

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

def get_name_of_allah_and_explanation(names_flag):
    if not names_flag:
        return None
    else:
        # Get the names of Allah
        names_of_allah = Project_Quran().Get_Names_of_Allah_English()

        # get the arabic names of Allah
        names_of_allah_arabic = Project_Quran().Get_Names_of_Allah_Arabic()

        # Get the transliterated names of Allah
        allah_transliterations = Project_Quran().Get_Names_of_Allah_Transliteration()

        # Select a index in the range of the number of names of Allah
        index = random.randint(0, len(names_of_allah) - 1)

        # Get the name of Allah in English
        name_of_allah_english = names_of_allah[index]

        # Get the name of Allah in Arabic
        name_of_allah_arabic = names_of_allah_arabic[index]

        # Get the transliteration of the name of Allah
        name_of_allah_transliteration = allah_transliterations[index]

        # Get the explanation of the name of Allah using a GPT-3 model prompt
        prompt = f"""

        For the name of Allah below, please do the following: First break down the word into its root letters, giving their approximate equivalent in the english language. Then, using the root letters, try to guess the meaning of the name. Finally, give a brief explanation of the significane of this particular name of Allah as it relates to the Quran.
        
        Name of Allah: {name_of_allah_arabic}\nExplanation: 
        
        """

        # Get the explanation of the name of Allah
        explanation = query_gpt(prompt)

        # print the name of the Allah and the explanation in a nice format
        print(f'\n\n\n\n ---------------- NAME OF ALLAH ---------------- \n')
        print(f'---------------- {name_of_allah_arabic}: {name_of_allah_transliteration} - {name_of_allah_english} ---------------- \n\n')
        print(f'Explanation: {explanation}\n\n\n\n')
        print(f'When you are ready to begin, press ENTER (wait 10 seconds).')
        input()
        time.sleep(10)

def get_verses_and_explanations(countdown_seconds):
    # Get the verses
    verses_Quran_Module, verses  = select_quran_verse()
    # print(f"Verse 1:\n{verses}")

    # Depending on the length of the countdown, select the number of verses to display
    if countdown_seconds < 3600: # less than 1 hour
        verses_Quran_Module = verses_Quran_Module[0:1] # only display the first verse
        verses = verses[0:1]
    elif countdown_seconds > 3600 and countdown_seconds < 7200: # between 1 and 2 hours
        verses_Quran_Module = verses_Quran_Module[0:2] # only display the first two verses
        verses = verses[0:2]
    else:
        pass

    # initialize the verse_texts output and explanations output
    verse_texts = []
    explanations = []

    # Get the explanations
    for i, verse_info in enumerate(zip(verses_Quran_Module, verses)):
        verse_QM, verse = verse_info
        verse_text = Project_Quran().Get_Ayah_English(verse_QM).split('"')[1][0:-1]
        chapter_number = verse.split(':')[0]
        if i == 0:
            prompt2 = f"""
            I want you to act as Sheikh Hamza Yusuf. You are an influential Islamic scholar, who has a profound understanding of the Qur'an, Hadith, and various branches of Islamic Jurisprudence. 
            You are deeply conversant with the various cultural, historical, and philosophical contexts of Islamic thought. You are committed to promoting peace, understanding, and the intellectual tradition of Islam. 
            I know that you are just an AI, but I don't want you to say that at the beginning, just emulate this description to the best of your ability. My first request is as follows:

            First please give some brief context about Chapter {chapter_number} from the Qur'an. Then, to the best of your ability, explain the meaning of the verse below. 
            I understand you are just an AI language model, but please remember I don't want you to state or acknowledge that, just try to interpret the possible meaning of the verse given. \n\n Verse: {verse_text} \n\n Explanation:
            """
        else:
            prompt2 = f"""
            I want you to act as Sheikh Hamza Yusuf. You are an influential Islamic scholar, who has a profound understanding of the Qur'an, Hadith, and various branches of Islamic Jurisprudence. 
            You are deeply conversant with the various cultural, historical, and philosophical contexts of Islamic thought. You are committed to promoting peace, understanding, and the intellectual tradition of Islam. 
            I know that you are just an AI, but I don't want you to say that at the beginning, just emulate this description to the best of your ability. My first request is as follows:

            To the best of your ability, explain the meaning of the verse below. 
            I understand you are just an AI language model, but please remember I don't want you to state or acknowledge that, just try to emulate Sheikh Hamza Yusuf and interpret the possible meaning of the given verse. \n\n Verse: {verse_text} \n\n Explanation:
            """
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

