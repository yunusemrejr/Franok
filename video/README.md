# Franok, from scratch

The film is a 1280 × 720, 24 fps technical walkthrough with synthesized English narration. The seven scenes follow actual token IDs and a worked bank rotation. The loss graphic illustrates two chosen probabilities. It does not plot measured training loss.

The finished MP4, English and Turkish WebVTT files and poster are in [web/media](../web/media). To render from source, install Pillow and FFmpeg, make the Lato and DejaVu fonts available as in `render.py`, and use a locally licensed Piper English voice. The release was rendered with en_US-ryan-high and an installed Piper binary.

```sh
python video/audio.py --piper /path/to/piper --voice /path/to/voice.onnx
python video/render.py
```

Caption timing comes from each synthesized narration line. The renderer writes the audio, frame samples and final video into `work/video/`. Notebook users can play the release directly in the Architecture tab, including offline in the desktop package.

The narrator uses the [Piper Ryan high voice](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/ryan/high/MODEL_CARD). Its voice card attributes its training data to RyanSpeech under CC BY-NC-SA 4.0. The repository distributes the synthesized narration as part of the film. The speech model and its training recordings are separate from the Franok model and its MIT-licensed weights.
