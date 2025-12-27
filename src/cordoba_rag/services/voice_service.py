import os
from fastapi import HTTPException
import azure.cognitiveservices.speech as speechsdk


def stt_from_wav(path_wav: str) -> str:
    speech_key = os.environ.get("SPEECH_KEY")
    speech_region = os.environ.get("SPEECH_REGION")
    if not speech_key or not speech_region:
        raise HTTPException(status_code=500, detail="Faltan SPEECH_KEY o SPEECH_REGION en .env")

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = "es-ES"

    audio_config = speechsdk.audio.AudioConfig(filename=path_wav)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text

    if result.reason == speechsdk.ResultReason.NoMatch:
        raise HTTPException(status_code=400, detail="No se pudo reconocer voz en el audio (NoMatch).")

    details = speechsdk.CancellationDetails.from_result(result)
    raise HTTPException(status_code=500, detail=f"STT cancelado: {details.reason} - {details.error_details}")


def tts_to_wav(text: str, output_wav_path: str) -> None:
    speech_key = os.environ.get("SPEECH_KEY")
    speech_region = os.environ.get("SPEECH_REGION")
    if not speech_key or not speech_region:
        raise HTTPException(status_code=500, detail="Faltan SPEECH_KEY o SPEECH_REGION en .env")

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_synthesis_voice_name = "es-ES-ElviraNeural"
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
    )

    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_wav_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    result = synthesizer.speak_text_async(text).get()

    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        details = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
        raise HTTPException(status_code=500, detail=f"TTS cancelado: {details.reason} - {details.error_details}")
