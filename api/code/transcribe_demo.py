#! python3.9
#
#  Example usage:
#   $ python3 transcribe_demo.py --model small --default_microphone "USB PnP Audio Device: Audio (hw:2,0)" 2> /dev/null
#

import os
import sys

# Silenciar los mensajes de bienvenida de Pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from typing import Any
from gtts import gTTS
from time import sleep
from tempfile import NamedTemporaryFile
from queue import Queue
from datetime import datetime, timedelta
from colorama import Fore, Style, init
import torch
import whisper
import speech_recognition as sr
import io
import argparse
import threading
import pygame
import time

init(autoreset=True)

# Importar módulos personalizados
from lib.openai import OPENAI
openai_instance = OPENAI()


class Loader:
    def __init__(self, msg="Loading", sleep=0.5):
        self.stopThread = False
        self.sleep = sleep
        self.msg = msg
        self.thread = threading.Thread(target=self.loading)
        self.start()

    def start(self):
        if self.thread.is_alive():
            self.thread.join()
    
        self.thread.start()

    def stop(self):
        self.stopThread = True
        self.thread.join()

    def loading(self):
        first = True
        contador = 0
        while not self.stopThread:
            if contador % 3 == 0:
                if not first:
                    print('\033[F\033[K', end='')
                else:
                    first = False

                print(f"{Style.BRIGHT}{Fore.LIGHTCYAN_EX}{self.msg} .")
            elif contador % 3 == 1:
                print('\033[F\033[K', end='')
                print(f"{Style.BRIGHT}{Fore.LIGHTCYAN_EX}{self.msg} ..",)

            else:
                print('\033[F\033[K', end='')
                print(f"{Style.BRIGHT}{Fore.LIGHTCYAN_EX}{self.msg} ...")

            contador += 1

            time.sleep(self.sleep)

        print('\033[F\033[K', end='')


class AudioHandler:
    def __init__(self, energy_threshold=1000, record_timeout=20, default_microphone=None, audio_model='small'):
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = energy_threshold
        self.recorder.dynamic_energy_threshold = False
        self.last_sample = bytes()
        self.data_queue = Queue()
        self.audio_model = None
        self.temp_file = NamedTemporaryFile().name
        self.source = None
        self.recording_allowed = True
        self.record_timeout = record_timeout

        self.setup_microphone(default_microphone)

        loader = Loader("Load model")
        self.load_audio_model(audio_model)
        loader.stop()

    def setup_microphone(self, default_microphone=None):

        if 'linux' in sys.platform:
            if default_microphone is None:
                print(f"")
                print(
                    f"{Style.BRIGHT}{Fore.YELLOW}Available microphone devices are: ")
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    print(
                        f"\t{Style.BRIGHT}{Fore.WHITE}{index}:{Style.NORMAL}{Fore.YELLOW} Microphone with name \"{name}\" found")
                print(f"")
                print(
                    f"\t{Style.BRIGHT} --> Please enter the number of the desired microphone(0-{len(sr.Microphone.list_microphone_names())}): ", end='')
                mic_index = input()
                self.source = sr.Microphone(device_index=int(mic_index))
            else:
                self.source = None
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if default_microphone in name:
                        self.source = sr.Microphone(device_index=index)
                        break
                if self.source is None:
                    print(
                        f"Failed to assign audio device {default_microphone}. Check help for more info.")
                    exit(1)
        else:
            self.source = sr.Microphone(sample_rate=16000)

        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source, duration=2)

    def load_audio_model(self, model='small'):
        self.audio_model = whisper.load_model(model)

    def record_callback(self, _, audio: sr.AudioData):
        if self.recording_allowed and not self.data_queue.full():
            data = audio.get_raw_data()
            self.data_queue.put(data)

    def listen_in_background(self):
        self.recorder.listen_in_background(
            self.source,
            self.record_callback,
            phrase_time_limit=self.record_timeout
        )

    def checking(self):
        loader_listening = Loader("Listening")
        while True:
            try:
                if not self.data_queue.empty():
                    loader_listening.stop()
                    loader = Loader("Processing audio")
                    text = self.process_audio()
                    loader.stop()

                    if text.lower() in ["adios", "hasta luego", "adiós.", "hasta luego!"]:
                        self.speak("Hasta luego!")
                        break
                    elif text != "":
                        self.recording_allowed = False

                        try:
                            print(
                                f"{Style.BRIGHT}{Fore.LIGHTGREEN_EX}YO: {Fore.WHITE}{text}")
                            loader = Loader("Generating response")
                            response = openai_instance.ChatCompletion(text)
                            loader.stop()
                            self.speak(response)
                            loader = Loader("Listening")
                        except Exception as e:
                            print(
                                f"Error during transcription or interaction: {e}")
                            break

                        self.recording_allowed = True

                    while not self.data_queue.empty():
                        self.data_queue.get()
                        self.data_queue.task_done()

                    text = ""
                    self.last_sample = bytes()

                    sleep(0.25)
            except KeyboardInterrupt:
                break

    def process_audio(self):
        while not self.data_queue.empty():
            data = self.data_queue.get()
            self.last_sample += data

        audio_data = sr.AudioData(
            self.last_sample,
            self.source.SAMPLE_RATE,
            self.source.SAMPLE_WIDTH
        )
        wav_data = io.BytesIO(audio_data.get_wav_data())
        with open(self.temp_file, 'w+b') as f:
            f.write(wav_data.read())

        result = self.audio_model.transcribe(
            self.temp_file,
            fp16=torch.cuda.is_available()
        )
        text = result['text'].strip()
        return text

    def speak(self, text):
        print(f"{Style.BRIGHT}{Fore.LIGHTMAGENTA_EX}NEREA: {Fore.WHITE}{text}")

        loader = Loader("Generating speaking audio")
        tts = gTTS(text=text, lang='es')
        audio_file = 'audio_stt.mp3'
        tts.save(audio_file)
        loader.stop()

        loader = Loader("Nerea is talking")
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        os.remove(audio_file)
        loader.stop()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="small",
                        help="Model to use", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=20,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--default_microphone", default=None,
                        help="Default microphone name for SpeechRecognition.", type=str)

    return parser.parse_args()


def script_title():
    title = f"""{Fore.MAGENTA}{Style.BRIGHT}
 .-----------------. .----------------.  .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
| | ____  _____  | || |  _________   | || |  _______     | || |  _________   | || |      __      | |
| ||_   \|_   _| | || | |_   ___  |  | || | |_   __ \    | || | |_   ___  |  | || |     /  \     | |
| |  |   \ | |   | || |   | |_  \_|  | || |   | |__) |   | || |   | |_  \_|  | || |    / /\ \    | |
| |  | |\ \| |   | || |   |  _|  _   | || |   |  __ /    | || |   |  _|  _   | || |   / ____ \   | |
| | _| |_\   |_  | || |  _| |___/ |  | || |  _| |  \ \_  | || |  _| |___/ |  | || | _/ /    \ \_ | |
| ||_____|\____| | || | |_________|  | || | |____| |___| | || | |_________|  | || ||____|  |____|| |
| |              | || |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------'  '----------------' 
"""
    print(title)

def welcome(audio_handler):
    audio_handler.speak(
        "Hola, soy Nerea, tu asistente virtual. ¿En qué puedo ayudarte?"
    )

def main():
    args = parse_args()

    script_title()

    audio_handler = AudioHandler(
        energy_threshold=args.energy_threshold,
        record_timeout=args.record_timeout,
        default_microphone=args.default_microphone,
        audio_model=args.model
    )

    print(f"")
    print(f"{Style.BRIGHT}{Fore.WHITE}--------------------------------------------------------------")

    welcome(audio_handler)

    audio_handler.listen_in_background()

    audio_handler.checking()

    print(f"{Style.BRIGHT}{Fore.WHITE}--------------------------------------------------------------")


if __name__ == "__main__":
    main()
