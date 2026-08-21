"""
============================================================
IGBO ASR REAL AUDIO TEST
============================================================

PURPOSE:
Tests the NCAIR Igbo ASR model using a real recording of
someone speaking Igbo.

BEFORE RUNNING:
1. Download/save a real Igbo audio recording (WAV recommended).
2. Open the Colab notebook.
3. Copy this entire file into ONE Colab code cell and run it.
4. When prompted, upload the Igbo audio recording.
5. The model will transcribe the recording and display its
   actual ASR output.

NOTE:
The test uses actual spoken Igbo audio, and the displayed
output is the model's prediction. Repeat with different
recordings to test different speakers/audio.

============================================================
"""

# STEP 1:
# Open Colab's file upload window.
# Select the real Igbo audio recording you downloaded to your
# computer.
#
# The complete contents of this file should be run in ONE
# Colab code cell.

from google.colab import files

uploaded = files.upload()


# STEP 2:
# Get the name of the audio file that was uploaded.
audio_file = next(iter(uploaded.keys()))

print(f"Uploaded file: {audio_file}")


# STEP 3:
# Import the transcription function from our ASR module.
from asr.transcribe import transcribe_audio


# STEP 4:
# Send the uploaded audio recording to the NCAIR Igbo ASR model.
#
# "Igbo" tells transcribe_audio() to use:
# NCAIR1/Igbo-ASR

print("\nTranscribing Igbo recording...")

transcript = transcribe_audio(
    audio_file,
    "Igbo"
)


# STEP 5:
# Display the actual transcription produced by the NCAIR
# Igbo ASR model.

print("\n--- NCAIR IGBO ASR OUTPUT ---")
print(transcript)