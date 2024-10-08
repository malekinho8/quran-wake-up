import os
import sys; sys.path.append('./')
import click
import time
import openai
import queue
from threading import Thread, Event
from fajrGPT.utils import *

client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
check_ffmpeg()

@click.command()
@click.option('--countdown-time', required=True, help='Countdown time in format [number][h/m/s], i.e. 1h would create a 1 hour timer.')
@click.option('--surah', required=False, help='Specific Surah from the Quran to play for the alarm audio (int between 1 and 114). Default is the first chapter (Surah Al-Fatihah).', default=1)
@click.option('--surah-verse-start', required=False, help='Specific verse from the Surah to start the alarm audio from (int). Default is the first verse.', default=1)
@click.option('--surah-verse-end', required=False, help='Specific verse from the Surah to end the alarm audio at (int). Default is the last verse (-1).' , default=-1)
@click.option('--names-flag', required=False, help='Whether or not to include a randomly selected name of Allah in the preamble.', default=True)
@click.option('--english', required=False, help='Whether or not to play audio with the english translation of the Quran verses.', default=False)
@click.option('--low-pass', required=False, help='Amount of low-pass to apply to the audio (float (KHz) or None). Default is 10 (KHz).', default=10)
@click.option('--gpt-model-type', required=False, help='Which GPT model to use for the prompt responses from OpenAI.', default="gpt-4-0314")
@click.option('--telegraphic', required=False, help='Whether or not to use a telegraphic (i.e. very simple) speech style in the response.', default=True)
@click.option('--noise', required=False, default=False, help='Whether or not to play calming noise during the countdown.')
@click.option('--noise-type', required=False, default='brown', help='Type of noise to play during the countdown. Options are: brown, pink, white, blue, violet, grey, or red.')
@click.option('--max-noise-volume', required=False, default=1, help='Maximum volume of the noise audio.')
@click.option('--max-alarm-volume', required=False, default=0.04, help='Maximum volume of the alarm audio.')
@click.option('--scholar', required=False, default='hamza_yusuf', help='Which scholar to use for the Quran verse explanations. Options are: hamza_yusuf, nouman_ali_khan, or mufti_menk.')
@click.option('--override-number', required=False, default=None, help='Set the number of verses tp show in the explanation. Default is None, which will set the verse number selection process automatically based on the countdown time.')

def main(countdown_time, surah=1, names_flag=True, english=False, low_pass=10, gpt_model_type="gpt-4o", telegraphic=True, noise=False, noise_type='brown', max_noise_volume=1, max_alarm_volume=0.04, scholar='hamza_yusuf', override_number=None, surah_verse_start=1, surah_verse_end=-1):
    # initialize the result queues
    allah_queue = queue.Queue() if names_flag else None
    selected_verses_queue = queue.Queue()
    verses_explanations_queue = queue.Queue()
    quran_audio_queue = queue.Queue()
    alarm_out_queue = queue.Queue()

    # convert time to seconds
    countdown_seconds = convert_to_seconds(countdown_time)

    # Create threads for audio processing and countdown
    prepare_alarm_audio_thread = Thread(target=alarm_audio_processing, args=(surah, english, low_pass, surah_verse_start, surah_verse_end, alarm_out_queue))
    countdown_thread = Thread(target=countdown, args=(countdown_seconds,))
    countdown_stopper_thread = Thread(target=premature_countdown_stop, args=())

    # Create a thread for playing some noise during the countdown if the user has specified so
    if noise:
        play_noise_thread = Thread(target=play_noise, args=(noise_type,10000,0.5,max_noise_volume,))
        quran_fadein_time = 60 # seconds
    else:
        max_alarm_volume = 1.0
        quran_fadein_time = 900 # seconds

    # create threads for obtaining quran verse and Allah name explanations
    selected_verses_thread = Thread(target=select_quran_verse,args=(selected_verses_queue,))
    get_name_of_allah_thread = Thread(target=get_name_of_allah_and_explanation, args=(gpt_model_type,allah_queue,telegraphic)) if names_flag else None

    # Start the threads
    countdown_thread.start()
    countdown_stopper_thread.start()
    selected_verses_thread.start()

    # wait a second and then start the noise thread if it was specified
    if noise:
        time.sleep(1)
        play_noise_thread.start()
    
    # wait for the selected verse thread to finish
    selected_verses_thread.join()

    # fetch the results from the queue
    verses_Quran_Module, selected_verses = selected_verses_queue.get()

    # for the selected verses, get the explanations and corresponding audio on a separate thread
    get_explanations_thread = Thread(target=get_explanations, args=(verses_Quran_Module,selected_verses,countdown_seconds,gpt_model_type,verses_explanations_queue,telegraphic,scholar,override_number))
    prepare_selected_verse_audio_thread = Thread(target=download_quran_verses_audio,args=(selected_verses,quran_audio_queue,))

    # wait a second before starting the explanations and audio downloading threads while the countdown continues
    time.sleep(1)
    get_name_of_allah_thread.start() if names_flag else None
    get_explanations_thread.start()
    prepare_alarm_audio_thread.start()
    prepare_selected_verse_audio_thread.start()

    # Wait for both threads to finish
    countdown_thread.join()

    # stop the countdown stopper thread
    countdown_stopper_thread.join()

    # stop the noise thread if it was started
    if noise:
        set_global_stop_audio_flag()
        play_noise_thread.join()
    
    # start playing the alarm audio
    prepare_alarm_audio_thread.join() # join the thread.
    selected_quran_audio_file = quran_audio_queue.get()[0]
    alarm_output_file = alarm_out_queue.get()[0] + '.mp3'

    # process the selected quran audio file on a separate thread
    filter_selected_audio_thread = Thread(target=apply_low_pass_filter,args=( selected_quran_audio_file, float(low_pass * 1000) ))
    filter_selected_audio_thread.start()
    filter_selected_audio_thread.join()

    # Play alarm audio with fade-in effect on a separate thread
    play_audio_thread = Thread(target=play_audio, args=(alarm_output_file, quran_fadein_time, max_alarm_volume,))
    play_audio_thread.start()

    # wait for the explanations threads to finish. if there are any errors the code will hang here after the countdown has finished and noise has stopped.
    get_name_of_allah_thread.join() if names_flag else None
    get_explanations_thread.join()

    # fetch the gpt results from the queue
    name_of_allah_arabic, name_of_allah_transliteration, name_of_allah_english, explanation = allah_queue.get() if names_flag else None
    verse_texts, explanations, verses = verses_explanations_queue.get()

    # display the name of Allah and explanation
    display_allah_name_and_explanation(name_of_allah_arabic, name_of_allah_transliteration, name_of_allah_english, explanation) if names_flag else None

    print(f'When you are ready to see the selected Quran verses, press ENTER.')
    input()

    # print the selected verses
    print_selected_verses(verses)

    # stop the misharay audio once the user has finished reading the name of Allah
    stop_audio(5,)

    # play the audio of the quran verses with 5 second fade in
    play_audio_thread = Thread(target=play_audio, args=(selected_quran_audio_file, quran_fadein_time, max_alarm_volume,))
    play_audio_thread.start()

    # display the explanations
    display_quran_verse_explanations(verse_texts,explanations,verses)

    # stop the audio once the user has completed reading the verses
    stop_audio(5,)

    # return back to the main thread
    play_audio_thread.join()

if __name__ == "__main__":
    main()

