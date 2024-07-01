import os
import sys; sys.path.append('./')
import pygame
import numpy as np
import time
import subprocess
import requests
import openai
import random
import tempfile
import mutagen.mp3 as mp3
import math
import pkg_resources
import tempfile
import selectors
from threading import Event
from tqdm import tqdm
from Quran_Module import Project_Quran
from scipy.signal import butter, lfilter
from pydub import AudioSegment
from fajrGPT.quran_metadata import quran_chapter_to_verse, surah_number_to_name_tag

# Declare global variables
bypass_countdown_flag = False
countdown_finished = False
countdown_finished_event = Event()

def play_noise(noise_type, crossfade_duration=2000, crossfade_point=0.1666, audio_length=120, max_volume=1):
    # set the file path depending on the noise type given
    try:
        # Use resource_stream to access the file
        temp_file_path = access_package_file('fajrGPT', f'assets/noise/{noise_type}.mp3') 
    except:
        raise Exception(f'Error: {noise_type} is not supported (yet).')
    # being audio playing loop
    play_audio_loop(temp_file_path, crossfade_duration, crossfade_point, max_volume)
    os.remove(temp_file_path)

def access_package_file(package_name, file_path):
    # Use resource_stream to access the file
    stream = pkg_resources.resource_stream(package_name, file_path)

    # Create a temporary file and write the stream to it
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(stream.read())
        temp_file_path = tmp_file.name

    return temp_file_path

def access_and_read_package_file(package_name, file_path):
    # Use resource_stream to access the file
    stream = pkg_resources.resource_stream(package_name, file_path)

    # Read the file
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        tmp_file.write(stream.read())
        tmp_file.seek(0)
        temp_file_path = tmp_file.name
        with open(temp_file_path, 'r') as f:
            text = f.read()

    return text

def print_selected_verses(verses:list):
    # print the verses selected
    print(f'\n\nQuran Verses Selected:\n')
    for verse in verses:
        print(f'{verse}')

def alarm_audio_processing(surah, english, low_pass, surah_verse_start, surah_verse_end, out_queue=None):
    # make the output file name the same as the surah
    output_folder = 'quran-audio'
    output_name = 'surah-' + '{:03d}'.format(int(surah)) + '-verse-' + '{:03d}'.format(int(surah_verse_start)) + '{:03d}'.format(int(surah_verse_end))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    output = os.path.join(output_folder, output_name)
    if not english:
        flag = download_surah(surah, output, surah_verse_start, surah_verse_end)
    else:
        output += '-english'
        flag = download_surah_with_english(surah, output, surah_verse_start, surah_verse_end)

    # test audio
    test_audio(f'{output}.mp3', output, flag)

    # apply low pass filter to the audio
    if low_pass:
        cutoff_filter = float(low_pass) * 1000
        apply_low_pass_filter(f'{output}.mp3', cutoff_filter)
    
    if out_queue:
        out_queue.put([output])
    else:
        return True

def download_surah_with_english(surah, output, surah_verse_start, surah_verse_end):
    # check if the output file already exists
    if os.path.exists(f'{output}.mp3'):
        # if the output file exists, then return True
        return True
    else:
        # obtain the number of verses in the surah (+1 corresponds to the basmalah)
        surah_verse_end = quran_chapter_to_verse[int(surah)] + 1  if surah_verse_end == -1 else surah_verse_end
        # first obtain the english verses urls in the surah (+1 corresponds to the basmalah)
        urls = [quran_english_verse_to_mp3_url("1:1")] + [quran_english_verse_to_mp3_url(f'{surah}:{verse}') for verse in range(surah_verse_start,surah_verse_end)]
        # download the verse audio and sleep for 0.5 seconds between each download
        english_file_paths = [download_file_and_sleep(url,0.5) for url in urls]
        # then obtain the arabic verses urls in the surah (+1 corresponds to the basmalah)
        urls = [quran_verse_to_mp3_url("1:1")] + [quran_verse_to_mp3_url(f'{surah}:{verse}') for verse in range(surah_verse_start,surah_verse_end)]
        # download the verse audio and sleep for 0.5 seconds between each download
        arabic_file_paths = [download_file_and_sleep(url,0.5) for url in urls]
        # combine the file paths such that it goes arabic, english, arabic, english, etc.
        combined_file_paths = [val for pair in zip(arabic_file_paths, english_file_paths) for val in pair]
        # combine the audio files
        combined_audio = AudioSegment.empty()
        for file_path in combined_file_paths:
            combined_audio += AudioSegment.from_mp3(file_path)

        # save the combined audio file
        combined_audio.export(f'{output}.mp3', format="mp3")

        # return False
        return False

def download_surah(surah, output, surah_verse_start, surah_verse_end):
    # check if the output file already exists
    if os.path.exists(f'{output}.mp3'):
        # if the output file exists, then return True
        return True
    else:
        # obtain the number of verses in the surah (+1 corresponds to the basmalah)
        end_verse_number = quran_chapter_to_verse[int(surah)] + 1  if surah_verse_end == -1 else surah_verse_end
        
        # loop over all the verses in the surah and construct the urls
        urls = [quran_verse_to_mp3_url("1:1")] + [quran_verse_to_mp3_url(f'{surah}:{verse}') for verse in range(surah_verse_start,end_verse_number)]

        # download the verse audio and sleep for 0.5 seconds between each download
        file_paths = [download_file_and_sleep(url,0.5) for url in urls]

        # combine the audio files
        combined_audio = AudioSegment.empty()
        for file_path in file_paths:
            combined_audio += AudioSegment.from_mp3(file_path)
        
        # save the mp3 to the output file
        combined_audio.export(f'{output}.mp3', format="mp3")

        # delete the temporary files
        for file_path in file_paths:
            os.remove(file_path)

    return True    

def download_quran_verses_audio(verses, quran_audio_queue):
    # convert verses to urls
    urls = [quran_verse_to_mp3_url(verse) for verse in verses]

    # download the verse audio
    file_paths = [download_file_and_sleep(url,0.5) for url in urls]

    # concatenate the audio files into one
    combined_audio_file_name = combine_audio_from_files(file_paths)

    # delete the temporary files
    for file_path in file_paths:
        os.remove(file_path)
        
    quran_audio_queue.put([combined_audio_file_name])

def combine_audio_from_files(file_paths):
    # combine the audio files
    combined_audio = AudioSegment.empty()
    for file_path in file_paths:
        combined_audio += AudioSegment.from_mp3(file_path)

    # create a temporary file to store the combined audio
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)

    # export the combined audio to the temporary file
    combined_audio.export(temp_file.name, format='mp3')

    # return the path to the temporary file
    return temp_file.name

def quran_surah_with_english_to_mp3_url(surah):
    # convert surah to int and get the tag from the quran_chapter_to_tag dictionary
    surah = int(surah)
    tag = surah_number_to_name_tag[surah]

    # get the url of the mp3 file
    url = f'https://thechosenone.info/wp-content/uploads/2014/09/{tag}.mp3'

    return url

def quran_verse_to_mp3_url(verse):
    # get the chapter and verse number
    chapter, verse_number = verse.split(':')

    # convert the numbers to the correct string format
    chapter = chapter.zfill(3)
    verse_number = verse_number.zfill(3)

    # get the url of the mp3 file
    url = f'https://islamicstudies.info/quran/afasy/audio/{chapter}{verse_number}.mp3'

    return url

def quran_english_verse_to_mp3_url(verse):
    # get the chapter and verse number
    chapter, verse_number = verse.split(':')

    # convert the numbers to the correct string format
    chapter = chapter.zfill(3)
    verse_number = verse_number.zfill(3)

    # get the url of the mp3 file
    url = f'https://everyayah.com/data/English/Sahih_Intnl_Ibrahim_Walk_192kbps/{chapter}{verse_number}.mp3'

    return url

def create_looped_segment(audio_segment, num_loops, crossfade_duration):
    # Initialize the looped segment with the first iteration
    looped_segment = audio_segment

    # Loop and concatenate with crossfade
    for _ in range(1, num_loops):
        looped_segment = looped_segment.append(audio_segment, crossfade=crossfade_duration)

    return looped_segment

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
    # specify a global variable to store if the countdown has finished or not
    global countdown_finished
    countdown_finished = False
    global bypass_countdown_flag
    bypass_countdown_flag = False

    # use tqdm to display the countdown progress
    print('\n\n\n\n ---------------- BEGINNING COUNTDOWN ---------------- \n\n')
    # print to the user that they can stop the countdown by pressing Enter
    print('Press Enter to stop the countdown prematurely. \n\n\n')
    # print the current time in HH:MM format
    print(f'\n\nSTART TIME: {time.strftime("%H:%M", time.localtime())}\n\n')
    for i in tqdm(range(int(countdown_seconds))):
        time.sleep(1)
        if bypass_countdown_flag:
            break
    
    # set the countdown_finished flag to True
    countdown_finished = True
    # set the event to indicate that the countdown has finished
    countdown_finished_event.set()
    print('\n\n\n\n ---------------- COUNTDOWN COMPLETE ----------------')
    # print the current time in HH:MM format
    print(f'\n\nEND TIME: {time.strftime("%H:%M", time.localtime())}\n\n')

# define a function to stop the countdown prematurely if the user presses Enter
def premature_countdown_stop():
    global bypass_countdown_flag
    # Create a default selector object to monitor I/O events
    sel = selectors.DefaultSelector()
    # Register the standard input (stdin) for read events
    sel.register(sys.stdin, selectors.EVENT_READ)
    # Loop until the countdown finished event is set
    while not countdown_finished_event.is_set():
        # Use the selector to check for I/O events with a timeout
        events = sel.select(timeout=1)  # Adjust the timeout as needed
        # Iterate over the events
        for key, mask in events:
            # Check if the event is from stdin
            if key.fileobj == sys.stdin:
                # Read the user input from stdin and strip any extra whitespace
                user_input = sys.stdin.readline().strip()
                # If the user input is an empty string (Enter key pressed)
                if user_input == '':
                    # Set the bypass countdown flag to True
                    bypass_countdown_flag = True      
                    # Set the countdown finished event to stop the countdown thread
                    countdown_finished_event.set()       
                    # Exit the function
                    return

# define a function to open text files
def open_text_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return text

def get_name_of_allah_and_explanation(gpt_model_type, allah_queue,telegraphic_flag):
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
    telegraphic_prompt = access_and_read_package_file('fajrGPT', 'assets/prompt/telegraphic.txt') if telegraphic_flag else ""
    allah_prompt = access_and_read_package_file('fajrGPT', 'assets/prompt/allah_name_explanation.txt')
    prompt = allah_prompt + "\n\n" + telegraphic_prompt + "\n\n" + f"Name of Allah: {name_of_allah_arabic}\nExplanation:"

    # Get the explanation of the name of Allah
    explanation = query_gpt(prompt, gpt_model_type)

    # add the outputs to the queue
    allah_queue.put([name_of_allah_arabic, name_of_allah_transliteration, name_of_allah_english, explanation])

def display_allah_name_and_explanation(name_of_allah_arabic, name_of_allah_transliteration, name_of_allah_english, explanation):
    # print the name of the Allah and the explanation in a nice format
    print(f'\n\n\n\n ---------------- NAME OF ALLAH ---------------- \n')
    print(f'---------------- {name_of_allah_arabic}: {name_of_allah_transliteration} - {name_of_allah_english} ---------------- \n\n')
    print(f'Explanation: {explanation}\n\n\n\n')

def get_explanations(verses_Quran_Module,verses,countdown_seconds,gpt_model_type,out_queue,telegraphic_flag,scholar,override_number):
    # Depending on the length of the countdown, select the number of verses to display
    if override_number is None:
        if countdown_seconds < 3600: # less than 1 hour
            verses_Quran_Module = verses_Quran_Module[0:1] # only display the first verse
            verses = verses[0:1]
        elif countdown_seconds > 3600 and countdown_seconds < 7200: # between 1 and 2 hours
            verses_Quran_Module = verses_Quran_Module[0:2] # only display the first two verses
            verses = verses[0:2]
        else:
            pass
    else:
        verses_Quran_Module = verses_Quran_Module[0:int(override_number)]


    # initialize the verse_texts output and explanations output
    verse_texts = []
    explanations = []

    # access the scholar prompt
    scholar_prompt = access_and_read_package_file('fajrGPT', f'assets/prompt/scholars/{scholar}.txt')
    telegraphic_prompt = access_and_read_package_file('fajrGPT', 'assets/prompt/telegraphic.txt') if telegraphic_flag else ""

    # Get the explanations
    for i, verse_info in enumerate(zip(verses_Quran_Module, verses)):
        verse_QM, verse = verse_info
        verse_text = Project_Quran().Get_Ayah_English(verse_QM).split('"')[1][0:-1].replace("&quot;",'"').replace("&quot",'"')
        chapter_number = verse.split(':')[0]
        if i == 0:
            first_verse_prompt = access_and_read_package_file('fajrGPT', 'assets/prompt/first_verse_explanation.txt').format(chapter_number=chapter_number,telegraphic_prompt=telegraphic_prompt,verse_text=verse_text)
            quran_prompt = scholar_prompt + "\n\n" + first_verse_prompt
            explanation = query_gpt(quran_prompt,gpt_model_type)
        else:
            context = explanations[0].split('\n')[0] # get the context about what the chapter is about.
            verse_explanation_prompt = access_and_read_package_file('fajrGPT', 'assets/prompt/verse_explanation.txt').format(telegraphic_prompt=telegraphic_prompt,verse_text=verse_text,context=context)
            quran_prompt = scholar_prompt + "\n\n" + verse_explanation_prompt
            explanation = query_gpt(quran_prompt,gpt_model_type)
        verse_texts.append(verse_text)
        explanations.append(explanation)

    if not out_queue:
        return verse_texts, explanations, verses
    else:
        out_queue.put([verse_texts,explanations,verses])

def display_quran_verse_explanations(verse_texts, explanations, verses):
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

def select_quran_verse(output_queue=None):
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
    if not output_queue:
        return (f'59,{surah},{verse}', f'59,{surah},{verse + 1}', f'59,{surah},{verse + 2}'), (f'{surah}:{verse}',f'{surah}:{verse+1}' ,f'{surah}:{verse+2}') # 59 corresponds to english Quran verse
    else:
        verses_Quran_Module, verses = (f'59,{surah},{verse}', f'59,{surah},{verse + 1}', f'59,{surah},{verse + 2}'), (f'{surah}:{verse}',f'{surah}:{verse+1}' ,f'{surah}:{verse+2}') # 59 corresponds to english Quran verse
        output_queue.put([verses_Quran_Module, verses])

def query_gpt(prompt,gpt_model_type):
    # GPT-3 API parameters
    COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.4 because it gives the most predictable, factual answer with some variation.
    "temperature": 0.4,
    "max_tokens": 1000,
    "model": gpt_model_type,
    }
    
    response = openai.ChatCompletion.create(
                messages = [{"role": "user", "content": prompt}],
                **COMPLETIONS_API_PARAMS
            )

    return response['choices'][0]['message']['content']

def gradually_change_volume(start_volume, end_volume, duration):
    # specify a global variable to indicate whether or not the audio has finished changing volume
    global finished_changing_volume
    finished_changing_volume = False

    # Compute the number of steps and the change in volume per step
    steps = int(duration * 100)  # Multiply by 100 to allow for hundredths of seconds
    delay = 0.01  # Delay for each step

    max_steps = 10000  # Maximum number of steps

    # If steps are more than maximum, adjust steps and delay
    if steps > max_steps:
        steps = max_steps
        delay = duration / max_steps

    delta_volume = (end_volume - start_volume) / steps

    # Set the initial volume
    pygame.mixer.music.set_volume(start_volume)

    # Gradually change the volume
    for i in range(steps):
        # Wait for the calculated delay
        time.sleep(delay)

        # Change the volume
        new_volume = start_volume + i * delta_volume
        pygame.mixer.music.set_volume(new_volume)

        # If the stop_audio function has been called, break the loop
        if stop_audio_called:
            print("Stopping volume change due to stop_audio being called")
            break
    
def crossfade(c1, c2, fade_duration, max_volume=1):
    """
    Performs a crossfade between two channels over a specified duration.
    """
    global stop_audio_called
    steps = 200
    volume_step = max_volume / steps
    step_duration = fade_duration / steps

    total_volume = 10 * math.log10(10 ** (c1.get_volume() / 10) + 10 ** (c2.get_volume() / 10))

    for step in range(steps):
        if stop_audio_called:
            c1.fadeout(1000)
            c2.fadeout(1000)
            return
        
        c2.set_volume(step * volume_step)
        c1_volume = min(-1.073 * c2.get_volume()**2 + 0.18*c2.get_volume() + 1, 1)  
        # use the equation: x_{1} = -1.073 x_{2}^2 + 0.18x_2 + 0.95 determined experimentally
        c1.set_volume(c1_volume)
        time.sleep(step_duration / 1000)

    c2.set_volume(max_volume)
    c1.stop()

def play_audio_loop(mp3_path, fade_duration=2000, crossfade_point=0.166, max_volume=1):
    """
    Plays an audio loop with crossfade using two channels. The loop continues until stop_audio is set to True.
    """
    # Initialize Pygame mixer
    pygame.mixer.init()
    audio = pygame.mixer.Sound(mp3_path)
    audio_length_ms = audio.get_length() * 1000  # Length of the audio in milliseconds

    global stop_audio_called
    stop_audio_called = False
    channel1 = pygame.mixer.Channel(1)
    channel2 = pygame.mixer.Channel(2)

    # Set the initial volume of channel 2 to 0 and start playing channel 1
    channel2.set_volume(0)
    channel1.set_volume(0)
    channel1.play(audio, loops=0)
    current_channel = 1
    fade_in_channel(channel1, fade_duration, max_volume)

    # Begin a timer to track playback time
    start_time = time.time()

    while not stop_audio_called:
        if current_channel == 1 and (time.time() - start_time) * 1000 > audio_length_ms * crossfade_point:
            channel2.set_volume(0)
            channel2.play(audio, loops=0)
            crossfade(channel1, channel2, fade_duration)
            current_channel = 2
            start_time = time.time()
        elif current_channel == 2 and (time.time() - start_time) * 1000 > audio_length_ms * crossfade_point:
            channel1.set_volume(0)
            channel1.play(audio, loops=0)
            crossfade(channel2, channel1, fade_duration)
            current_channel = 1
            start_time = time.time()

        time.sleep(1)
    
    channel1.fadeout(fade_duration)
    channel2.fadeout(fade_duration)
    time.sleep(fade_duration/1000)
    channel1.stop()
    channel2.stop()
    
def fade_in_channel(channel, fade_duration, max_volume=1):
    """
    Gradually increases the volume of a channel to maximum over a specified duration.
    """
    global stop_audio_called
    steps = 200
    volume_step = max_volume / steps
    step_duration = fade_duration / steps

    for step in range(steps):
        if stop_audio_called:
            channel.fadeout(1000)
            return

        channel.set_volume(step * volume_step)
        time.sleep(step_duration / 1000)
    
    channel.set_volume(max_volume)


def play_audio(file_path_or_url, transition_time=900, max_volume=1):
    # only start playing the audio once the countdown has finished OR if the user has specified a special flag to be true
    while not countdown_finished and not bypass_countdown_flag:
        time.sleep(1)

    global stop_audio_called
    stop_audio_called = False
    file_path = ''

    # Check if file_path_or_url is URL
    if file_path_or_url.startswith('http'):
        # Download the file and get the path
        file_path = download_file(file_path_or_url)
    else:
        file_path = file_path_or_url

    # Initialize pygame mixer
    pygame.mixer.init()

    # Load the audio file
    pygame.mixer.music.load(file_path)

    # Start playing the audio with volume 0
    pygame.mixer.music.set_volume(0.0)
    pygame.mixer.music.play()

    # if the transition time is longer than half of the audio, set it to half of the audio length
    length_in_seconds = mp3.MP3(file_path).info.length
    if transition_time > 0.5 * length_in_seconds:
        transition_time = length_in_seconds * 0.5

    # Gradually increase the volume over 15 minutes in a separate thread
    gradually_change_volume(0.0, max_volume, transition_time)

    # Loop the audio until the stop_audio function is called
    while not stop_audio_called:
        if not pygame.mixer.music.get_busy():
            # The music has finished, restart it
            pygame.mixer.music.play()
        time.sleep(1)

    # Delete the file if it was downloaded
    if file_path_or_url.startswith('http'):
        os.remove(file_path)

def set_global_stop_audio_flag():
    global stop_audio_called
    stop_audio_called = True

def stop_audio(transition_time=10):
    global stop_audio_called
    stop_audio_called = True # stop the audio in the other thread
    time.sleep(1)
    stop_audio_called = False # reset back to False to allow volume to decrease
    gradually_change_volume(pygame.mixer.music.get_volume(), 0.0, transition_time)

    # Stop the audio completely
    stop_audio_called = True
    pygame.mixer.music.stop()

def download_file(url):
    """Download file from URL and return the path to the file"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers, stream=True)
    file = tempfile.NamedTemporaryFile(delete=False)
    file_path = file.name + ".mp3"
    with open(file_path, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=1024):
            fd.write(chunk)
    return file_path

def download_file_and_sleep(url, time_to_sleep=0.5):
    result = download_file(url)
    time.sleep(time_to_sleep)
    return result

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def array_to_audio_segment(np_array, audio):
    # Scale numpy array and convert to a list of integers
    int_array = np.int16(np_array * 2**15)
    return audio._spawn(int_array.tobytes())

def apply_low_pass_filter(mp3_filename, cutoff_filter, order=5, fs=44100):
    # Read the mp3 file
    audio = AudioSegment.from_mp3(mp3_filename)

    # Convert to numpy array
    np_array = audio.get_array_of_samples()

    # divide by max value to normalize the data
    np_array = np_array / np.max(np.abs(np_array))

    # Apply the low pass filter
    filtered = butter_lowpass_filter(np_array, cutoff_filter, fs, order)

    # Convert the numpy array back to an audio segment
    filtered_audio = array_to_audio_segment(filtered, audio)

    # Export the audio segment as an mp3 file
    filtered_audio.export(mp3_filename, format="mp3")

def check_ffmpeg():
    try:
        # Try to get the FFmpeg version
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError:
        raise SystemExit("FFmpeg is not installed or is not available in the system's PATH. "
                         "Please refer to the installation instructions at: "
                         "https://github.com/malekinho8/quran-wake-up/blob/main/README.md")