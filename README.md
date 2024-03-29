# fajrGPT

---

This Python Command-Line application assists in helping you wake up for Fajr prayer by providing automatically selected verses + explanations from the Quran after a set alarm time along with a soothing Islamic prayer fade-in and fade-out from [Mishari bin Rashid Alafasy](https://en.wikipedia.org/wiki/Mishari_bin_Rashid_Alafasy). The program uses the [Quran Module](https://github.com/UBISOFT-1/Quran_Module) to obtain Quran verses in English as well as OpenAI's API GPT models to produce corresponding Tafsir (explanation).

## Features

- Plays Alafsay's Quran recitation audio for user-selected Surah (default is Surah Al-Fathiha) from [Tahfeem](https://islamicstudies.info/tafheem.php)
- Includes option to play Alafsay's Quranic recitation with its corresponding English translation obtained from [The Chosen One](https://thechosenone.info/al-quran-with-english-translation/)
- Plays the downloaded audio after a user-specified countdown time (i.e. 10s, 15m, 7h), with volume gradually fading in.
- Option to select and display a Name of Allah and provide corresponding explanation of its meaning with the GPT API
- Displays automatically selected verses from the Quran and provides corresponding explanations.
- Depending on the length of the countdown, one, two, or three verses will be selected for display
- After each explanation, the program waits for the user to press Enter before proceeding.
- Once all explanations are provided, the audio volume gradually fades out over a 5-second period.

## Usage

### Installing FFmpeg

FFmpeg is a necessary dependency for this package in order to play audio mp3 files. Follow the instructions below based on your operating system:

#### Windows

##### *Option 1: Using Conda (recommended for Conda users)*
If you have Anaconda or Miniconda installed, you can install FFmpeg using the following command:

```
conda install -c conda-forge ffmpeg
```

##### *Option 2: Manual Installation*
1. Go to the FFmpeg official [download page](https://ffmpeg.org/download.html) and download the Windows version.
2. Extract the ZIP archive.
3. Add the path to the `bin` directory (where `ffmpeg.exe` is located) to your system's PATH variable.

#### macOS

Using Homebrew (recommended):

```
brew install ffmpeg
```

Or follow the manual installation process similar to Windows by downloading from the FFmpeg official site.

#### Linux (e.g., Ubuntu)

Using the package manager:

```
sudo apt install ffmpeg
```

For other Linux distributions, adjust the command according to your package manager.

### CLI Installation with PyPI Distribution

1. Clone repo and install packages (for developers):
    ```bash
    git clone https://github.com/malekinho8/quran-wake-up.git
    pip install -r requirements.txt
    pip install .
    ```
2. Alternatively, install one dependency manually and `fajrGPT` through PyPI:
    ```python
    pip install git+https://github.com/malekinho8/quran-module.git fajrGPT
    ```
3. Run the main.py script, passing the required parameters:
    ```bash
    fajrGPT --countdown-time <Seconds, Minutes, or Hours until Alarm> --surah <Which Surah (number) to play for the alarm, default is 1>
    ```

## Required Parameters

- `--countdown-time`: The amount of time 

## Dependencies

- Python 3.x
- `click`
- `pygame`
- `pydub`
- `moviepy`
- `openai`
- `scipy`
- `mutagen`
- `TheQuranModule`

## Notes

- The OpenAI API key must be set in an environment variable named `OPENAI_API_KEY`. See instructions [here](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwistt_z6Pb-AhXYFlkFHWN4DOwQFnoECBIQAw&url=https%3A%2F%2Fwww.immersivelimit.com%2Ftutorials%2Fadding-your-openai-api-key-to-system-environment-variables&usg=AOvVaw1gmVeeQmPOcDRJvPQNXdI6) for more details.

## License

[MIT](https://choosealicense.com/licenses/mit/)
