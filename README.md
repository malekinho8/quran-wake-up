# fajrGPT

---

This Python application assists in helping you wake up for Fajr prayer by providing 3 interactive verses + explanations from the Quran followed by a soothing Islamic prayer fade-in and fade-out [audio file from YouTube](https://www.youtube.com/watch?v=zlOKoHk9W0I). The program uses OpenAI's GPT-3.5 Turbo model to produce verses and corrrsponding Tafsir (explanation).

## Features

- Downloads audio from a provided YouTube URL.
- Plays the downloaded audio after a user-specified countdown time (hours), with volume gradually increasing over a 15-minute period.
- Displays verses from the Quran and provides corresponding explanations.
- After each explanation, the program waits for the user to press Enter before proceeding.
- Once all explanations are provided, the audio volume gradually decreases over a 5-second period.

## Usage

1. Clone repo and install packages (for developers):
    ```bash
    git clone https://github.com/malekinho8/quran-wake-up.git
    pip install -r requirements.txt
    pip install .
    ```
2. Alternatively, install one dependency manually and `fajrGPT` through PyPI:
    ```python
    pip install git+https://github.com/ytdl-org/youtube-dl.git fajrGPT
    ```
3. Run the main.py script, passing the required parameters:
    ```bash
    fajrGPT --url <YouTube URL> --hours <Hours for Countdown> --output <Output filename>
    ```
    Replace `<YouTube URL>`, `<Hours for Countdown>`, and `<Output filename>` with appropriate values.

## Required Parameters

- `--url`: The URL of the YouTube video to download the audio from.
- `--hours`: The countdown hours before the YouTube audio will play.
- `--output`: The name of the output file (audio).

## Dependencies

- Python 3.8 or later.
- `click`
- `youtube_dl`
- `pygame`
- `pydub`
- `moviepy`
- `openai`

## Notes

- The OpenAI API key must be set in an environment variable named `OPENAI_API_KEY`. See instructions [here](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwistt_z6Pb-AhXYFlkFHWN4DOwQFnoECBIQAw&url=https%3A%2F%2Fwww.immersivelimit.com%2Ftutorials%2Fadding-your-openai-api-key-to-system-environment-variables&usg=AOvVaw1gmVeeQmPOcDRJvPQNXdI6) for more details.

## License

[MIT](https://choosealicense.com/licenses/mit/)
