import os
import azure.cognitiveservices.speech as speechsdk

def main():
    speech_key = os.environ["SPEECH_KEY"]
    speech_region = os.environ["SPEECH_REGION"]

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_synthesis_voice_name = "es-ES-ElviraNeural"

    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    text = "Hola, soy tu asistente turístico de Córdoba."
    result = synthesizer.speak_text_async(text).get()

    print("Reason:", result.reason)

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("✅ TTS OK")
    else:
        # Si falla, imprime detalles útiles
        details = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
        print("❌ TTS FAIL")
        print("Cancellation reason:", details.reason)
        print("Error details:", details.error_details)

if __name__ == "__main__":
    main()
