# quran-wake-up

---

# README

## Quranic Recitation Assistant

This Python application assists in reciting verses from the Quran with explanations, followed by a soothing background music fade-in and fade-out effect. The program uses OpenAI's GPT-3.5 Turbo model to provide explanations for selected verses.

## Features

- Downloads audio from a provided YouTube URL.
- Plays the downloaded audio, with volume gradually increasing over a 15-minute period.
- Displays verses from the Quran and provides corresponding explanations.
- After each explanation, the program waits for the user to press the space bar before proceeding.
- Once all explanations are provided, the audio volume gradually decreases over a 5-second period.

## Usage

1. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the main.py script, passing the required parameters:
    ```bash
    python main.py --url <YouTube URL> --hours <Hours for Countdown> --output <Output filename>
    ```
    Replace `<YouTube URL>`, `<Hours for Countdown>`, and `<Output filename>` with appropriate values.

## Required Parameters

- `--url`: The URL of the YouTube video to download the audio from.
- `--hours`: The countdown hours before the program starts.
- `--output`: The name of the output file (audio).

## Dependencies

- Python 3.8 or later.
- `click`
- `youtube_dl`
- `pygame`
- `pydub`
- `moviepy`
- `threading`
- `quran_metadata`
- `time`
- `subprocess`
- `openai`
- `keyboard`

## Notes

- The OpenAI API key must be set in an environment variable named `OPENAI_API_KEY`.
- The script must be run with administrative privileges to correctly capture keyboard input.

## License

[MIT](https://choosealicense.com/licenses/mit/)
