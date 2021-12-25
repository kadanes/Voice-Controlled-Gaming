
import pyaudio
import wave

import sounddevice as sd
import soundfile as sf
import numpy as np
import scipy.io.wavfile as wav


# https://stackoverflow.com/questions/35344649/reading-input-sound-signal-using-python

def recordCommandPyAudio(duration=3, playback=False):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = duration
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    p.terminate()
    
    # playback

    if playback:
        
        p = pyaudio.PyAudio()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'rb')

        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        data = wf.readframes(CHUNK)

        while len(data) > 0:
            stream.write(data)
            data = wf.readframes(CHUNK)

        stream.stop_stream()
        stream.close()

        p.terminate()

    return wf

def recordCommandSounddevice(duration=3, playback=False):
    fs=16000
    command = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    print("* recording")
    sd.wait()

    if playback:
        print("Audio recording complete , Play Audio")
        sd.play(command, fs)
        sd.wait()
    print("* done recording")
    sf.write("./output.wav", command, fs)

    return command