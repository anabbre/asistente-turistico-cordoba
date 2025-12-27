import os
import azure.cognitiveservices.speech as speechsdk

AUDIO_PATH = "data/audio/prueba.wav"

def main():
    speech_key = os.environ["SPEECH_KEY"]
    speech_region = os.environ["SPEECH_REGION"]

    if not os.path.exists(AUDIO_PATH):
        raise FileNotFoundError(f"No existe el archivo: {AUDIO_PATH}")

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = "es-ES"

    audio_config = speechsdk.audio.AudioConfig(filename=AUDIO_PATH)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print(f"🎧 Transcribiendo: {AUDIO_PATH}")
    result = recognizer.recognize_once_async().get()

    print("Reason:", result.reason)

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("✅ STT OK")
        print("Texto:", result.text)
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("❌ STT: No se pudo reconocer voz (NoMatch)")
        print("Detalles:", result.no_match_details)
    else:
        details = speechsdk.CancellationDetails.from_result(result)
        print("❌ STT CANCELLED")
        print("Motivo:", details.reason)
        print("Detalles:", details.error_details)

if __name__ == "__main__":
    main()
