# Musical_Instrument_Audio_Classification

## wavfiles

``wavfiles`` folder contains audios in .wav format of different musical instruments.

If we want to add more data to this folder we have to download from the Internet the audio in .mp3 format and, using FFmpeg converter, change file type to .wav using this command in the terminal:

``C:\ffmpeg\bin\ffmpeg -i audio_name.mp3 audio_name.wav``

Here is some useful information on how to install this converter: https://phoenixnap.com/kb/ffmpeg-windows

## clean.py

``clean.py`` script provides a set of tools for processing and cleaning the audio data.

It starts by splitting audios in ``wavfiles`` folder into smaller segments (1.0 seconds by default) and deleting those with no sound (a threshold can be specified to define what an audio with no sound is). Afterwards, the valid segments of audio are stored in ``clean`` folder divided again by musical instruments.

## models.py

This script contains the definition of the models we will train to try to perform a good classification of the audios. It contains some models from scratch and other pretrained models. Feel free to include new models in order to get a better accuracy in the classification.
