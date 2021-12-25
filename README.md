# Voice-Controlled-Gaming

Welcome to our [project](https://github.com/parthv21/Voice-Controlled-Gaming/) - Voice Commands for an immersive gaming experience

To run our project you will need

1. python 3.7
2. PIP Packages - sounddevice, tensorflow, numpy, keyboard
3. [mGBA emulator](https://mgba.io/downloads.html)
4. ROM of game you want to voice control

To run a mario game with our trained tensorflow model run the following command:

```
python runner.py mario tf
```

To run a tetris game with our trained tensorflow model run the following command:

```
python runner.py tetris tf
```

If you want to test a new game then you will have to edit `KeyLogging.py` and add a new function for key press you want to support. You can take a look at how to define these functions by looking at our `pressKeyMario(command)` or `pressKeyTetris(command)` methods.

We have tuned and trained the [conv](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/models.py#L207) and [light](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/models.py#L333) from [speech commands](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands) model by tensor flow. Their recognizer code has been modified to support the task of rapidly detecting single keyword over detecting keywords in a spoken stream.

The dataset used by this project is [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge) and our system supports the following commands:

```
["Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go"]
```

### Tips

- If running on Mac then please run from the default terminal.
