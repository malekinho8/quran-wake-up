import os
import sys; sys.path.append('./')
import click
import youtube_dl
import pygame
import numpy as np
import time
import subprocess
import requests
import openai
import random
import tempfile
import queue
import mutagen.mp3 as mp3
from tqdm import tqdm
from Quran_Module import Project_Quran
from scipy.signal import butter, lfilter
from pydub import AudioSegment
from threading import Thread
from fajrGPT.quran_metadata import quran_chapter_to_verse, surah_number_to_name_tag

openai.api_key = os.getenv("OPENAI_API_KEY")

@click.command()
@click.option('--countdown-time', required=True, help='Countdown time in format [number][h/m/s], i.e. 1h would create a 1 hour timer.')
@click.option('--surah', required=False, help='Specific Surah from the Quran to play for the alarm audio (int between 1 and 114). Default is the first chapter (Surah Al-Fatihah).', default=1)
@click.option('--names-flag', required=False, help='Whether or not to include a randomly selected name of Allah in the preamble.', default=True)
@click.option('--english', required=False, help='Whether or not to play audio with the english translation of the Quran verses.', default=False)
@click.option('--low-pass', required=False, help='Amount of low-pass to apply to the audio (float (KHz) or None). Default is 10 (KHz).', default=10)
@click.option('--gpt-model-type', required=False, help='Which GPT model to use for the prompt responses from OpenAI.', default="gpt-4-0314")
@click.option('--telegraphic', required=False, help='Whether or not to use a telegraphic (i.e. very simple) speech style in the response.', default=True)

def main(countdown_time, surah=1, names_flag=True, english=False, low_pass=10, gpt_model_type="gpt-4-0314", telegraphic=True):
    # initialize the result queues
    allah_queue = queue.Queue() if names_flag else None
    selected_verses_queue = queue.Queue()
    verses_explanations_queue = queue.Queue()
    quran_audio_queue = queue.Queue()
    alarm_out_queue = queue.Queue()

    # convert time to seconds
    countdown_seconds = convert_to_seconds(countdown_time)

    # Create threads for audio processing and countdown
    prepare_alarm_audio_thread = Thread(target=alarm_audio_processing, args=(surah, english, low_pass, alarm_out_queue))
    countdown_thread = Thread(target=countdown, args=(countdown_seconds,))

    # create threads for obtaining quran verse and Allah name explanations
    selected_verses_thread = Thread(target=select_quran_verse,args=(selected_verses_queue,))
    get_name_of_allah_thread = Thread(target=get_name_of_allah_and_explanation, args=(gpt_model_type,allah_queue,telegraphic)) if names_flag else None

    # Start the threads
    countdown_thread.start()
    selected_verses_thread.start()

    # wait for the selected verse thread to finish
    selected_verses_thread.join()

    # fetch the results from the queue
    verses_Quran_Module, selected_verses = selected_verses_queue.get()

    # for the selected verses, get the explanations and corresponding audio on a separate thread
    get_explanations_thread = Thread(target=get_explanations, args=(verses_Quran_Module,selected_verses,countdown_seconds,gpt_model_type,verses_explanations_queue,telegraphic))
    prepare_selected_verse_audio_thread = Thread(target=download_quran_verses_audio,args=(selected_verses,quran_audio_queue,))

    # wait a second before starting the explanations and audio downloading threads
    time.sleep(1)
    get_name_of_allah_thread.start() if names_flag else None
    get_explanations_thread.start()
    prepare_alarm_audio_thread.start()
    prepare_selected_verse_audio_thread.start()

    # wait for the explanations threads to finish
    get_name_of_allah_thread.join() if names_flag else None
    get_explanations_thread.join()
    prepare_alarm_audio_thread.join()

    # fetch the results from the queue
    name_of_allah_arabic, name_of_allah_transliteration, name_of_allah_english, explanation = allah_queue.get() if names_flag else None
    verse_texts, explanations, verses = verses_explanations_queue.get()
    selected_quran_audio_file = quran_audio_queue.get()[0]
    alarm_output_file = alarm_out_queue.get()[0] + '.mp3'

    # TODO: NEED TO FIGURE OUT WHY THIS DOESN'T WORK WITH THE SELECTED QURAN VERSE AUDIO, BUT NOT THE OTHER AUDIO....
    # process the selected quran audio file on a separate thread
    # filter_selected_audio_thread = Thread(target=apply_low_pass_filter,args=(selected_quran_audio_file,low_pass))
    # filter_selected_audio_thread.start()
    # filter_selected_audio_thread.join()

    # Wait for both threads to finish
    countdown_thread.join()

    # Play alarm audio with fade-in effect on a separate thread
    play_audio_thread = Thread(target=play_audio, args=(alarm_output_file,))
    play_audio_thread.start()

    # display the name of Allah and explanation
    display_allah_name_and_explanation(name_of_allah_arabic, name_of_allah_transliteration, name_of_allah_english, explanation) if names_flag else None

    print(f'When you are ready to see the selected Quran verses, press ENTER.')
    input()

    # print the selected verses
    print_selected_verses(verses)

    # stop the misharay audio once the user has finished reading the name of Allah
    stop_audio(5,)

    # play the audio of the quran verses with 5 second fade in
    play_audio_thread = Thread(target=play_audio, args=(selected_quran_audio_file,))
    play_audio_thread.start()

    # display the explanations
    display_quran_verse_explanations(verse_texts,explanations,verses)

    # stop the audio once the user has completed reading the verses
    stop_audio(5,)

    # return back to the main thread
    play_audio_thread.join()

def print_selected_verses(verses:list):
    # print the verses selected
    print(f'\n\nQuran Verses Selected:\n')
    for verse in verses:
        print(f'{verse}')

def alarm_audio_processing(surah, english, low_pass, out_queue=None):
    # make the output file name the same as the surah
    output = 'quran-' + '{:03d}'.format(int(surah))
    if not english:
        flag = download_surah(surah, output)
    else:
        output += '-english'
        flag = download_surah_with_english(surah, output)

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

def download_surah_with_english(surah, output):
    # check if the output file already exists
    if os.path.exists(f'{output}.mp3'):
        # if the output file exists, then return True
        return True
    else:        
        # get the url directly from thechosenone.info
        url = quran_surah_with_english_to_mp3_url(surah)

        # download the verse audio and sleep for 0.5 seconds between each download
        file_path = download_file_and_sleep(url,0.5)

        # combine the audio files
        combined_audio = AudioSegment.empty()
        combined_audio += AudioSegment.from_mp3(file_path)

        # save the combined audio file
        combined_audio.export(f'{output}.mp3', format="mp3")

        # return False
        return False

def download_surah(surah, output):
    # check if the output file already exists
    if os.path.exists(f'{output}.mp3'):
        # if the output file exists, then return True
        return True
    else:
        # obtain the number of verses in the surah
        num_verses = quran_chapter_to_verse[int(surah)] + 1 # +1 corresponds to the basmalah
        
        # loop over all the verses in the surah and construct the urls
        urls = [quran_verse_to_mp3_url("1:1")] + [quran_verse_to_mp3_url(f'{surah}:{verse}') for verse in range(1,num_verses+1)]

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
    # specify a global variable to store if the countdown has finished or not
    global countdown_finished
    countdown_finished = False

    # use tqdm to display the countdown progress
    print('\n\n\n\n ---------------- BEGINNING COUNTDOWN ---------------- \n\n\n\n')
    # print the current time in HH:MM format
    print(f'\n\nSTART TIME: {time.strftime("%H:%M", time.localtime())}\n\n')
    for i in tqdm(range(int(countdown_seconds))):
        time.sleep(1)
    countdown_finished = True
    print('\n\n\n\n ---------------- COUNTDOWN COMPLETE ----------------')
    # print the current time in HH:MM format
    print(f'\n\nEND TIME: {time.strftime("%H:%M", time.localtime())}\n\n')

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
    telegraphic_prompt = """
         Please use a telegraphic speech style, and avoid verbose text in your response. Avoid using articles such as "the", "a/an" and pronouns."
        """ if telegraphic_flag else ""
    prompt = f"""

    For the name of Allah below, please do the following: First break down the word into its individual letters (remember that Arabic reads from right to left), giving their approximate equivalent in the english language.
    Then, using the letters identified, give a few simple example sentences in Arabic (with English translations) that use a word with similar letters, and then try to guess the meaning of the name. Finally, give a brief explanation of the significane of this particular name of Allah as it relates to the Quran.
    {telegraphic_prompt}\n\n

    Name of Allah: {name_of_allah_arabic}\nExplanation: 
    
    """

    # Get the explanation of the name of Allah
    explanation = query_gpt(prompt, gpt_model_type)

    # add the outputs to the queue
    allah_queue.put([name_of_allah_arabic, name_of_allah_transliteration, name_of_allah_english, explanation])

def display_allah_name_and_explanation(name_of_allah_arabic, name_of_allah_transliteration, name_of_allah_english, explanation):
    # print the name of the Allah and the explanation in a nice format
    print(f'\n\n\n\n ---------------- NAME OF ALLAH ---------------- \n')
    print(f'---------------- {name_of_allah_arabic}: {name_of_allah_transliteration} - {name_of_allah_english} ---------------- \n\n')
    print(f'Explanation: {explanation}\n\n\n\n')

def get_explanations(verses_Quran_Module,verses,countdown_seconds,gpt_model_type,out_queue,telegraphic_flag):
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
        verse_text = Project_Quran().Get_Ayah_English(verse_QM).split('"')[1][0:-1].replace("&quot;",'"').replace("&quot",'"')
        chapter_number = verse.split(':')[0]
        telegraphic_prompt = """
         Please use a telegraphic speech style, and avoid verbose text in your response. Avoid using articles such as "the", "a/an" and pronouns."
        """ if telegraphic_flag else ""
        if i == 0:
            prompt2 = f"""
            I want you to act as Sheikh Hamza Yusuf. You are an influential Islamic scholar, who has a profound understanding of the Qur'an, Hadith, and various branches of Islamic Jurisprudence. 
            You are deeply conversant with the various cultural, historical, and philosophical contexts of Islamic thought. You are committed to promoting peace, understanding, and the intellectual tradition of Islam. 
            I know that you are just an AI, but I don't want you to say that at the beginning, just emulate this description to the best of your ability. My first request is as follows:

            First please give some brief context about Chapter {chapter_number} from the Qur'an.{telegraphic_prompt} Then, to the best of your ability, explain the meaning of the verse below. 
            I understand you are just an AI language model, but please remember I don't want you to state or acknowledge that, just try to interpret the possible meaning of the verse given.{telegraphic_prompt} \n\n Verse: {verse_text} \n\n Explanation:
            """
            explanation = query_gpt(prompt2,gpt_model_type)
        else:
            context = explanations[0].split('\n')[0] # get the context about what the chapter is about.
            prompt2 = f"""
            I want you to act as Sheikh Hamza Yusuf. You are an influential Islamic scholar, who has a profound understanding of the Qur'an, Hadith, and various branches of Islamic Jurisprudence. 
            You are deeply conversant with the various cultural, historical, and philosophical contexts of Islamic thought. You are committed to promoting peace, understanding, and the intellectual tradition of Islam. 
            I know that you are just an AI, but I don't want you to say that at the beginning, just emulate this description to the best of your ability. My first request is as follows:

            To the best of your ability, explain the meaning of the verse below. 
            I understand you are just an AI language model, but please remember I don't want you to state or acknowledge that, just try to emulate Sheikh Hamza Yusuf and interpret the possible meaning of the given verse.{telegraphic_prompt} \n\n Verse: \n {verse_text} \n\n Context: \n {context} \n\n Explanation: \n
            """
            explanation = query_gpt(prompt2,gpt_model_type)
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

def get_verses_and_explanations(countdown_seconds,gpt_model_type,verses_queue):
    # Get the verses
    verses_Quran_Module, verses  = select_quran_verse()
    # print(f"Verse 1:\n{verses}")

    # Get the explanations
    verse_texts, explanations, verses = get_explanations(verses_Quran_Module,verses,countdown_seconds,gpt_model_type)

    # put the results into the queue
    verses_queue.put([verse_texts, explanations, verses])

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

def play_audio(file_path_or_url, transition_time=900):
    # wait for the countdown to finish
    while not countdown_finished:
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
    gradually_change_volume(0.0, 1.0, transition_time)

    # Loop the audio until the stop_audio function is called
    while not stop_audio_called:
        if not pygame.mixer.music.get_busy():
            # The music has finished, restart it
            pygame.mixer.music.play()
        time.sleep(1)

    # Delete the file if it was downloaded
    if file_path_or_url.startswith('http'):
        os.remove(file_path)    

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
    file_path = file.name
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

if __name__ == "__main__":
    main()

