import json
from flask import Flask, request, jsonify
import base64
import whisper
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
model = whisper.load_model("small")

def transcribe(audio):
    
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    # _, probs = model.detect_language(mel)

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False,language="hi")
    result = whisper.decode(model, mel, options)
    return result.text

app = Flask(__name__)

@app.route('/hello/', methods=['POST'])
def welcome():
    record = json.loads(request.data)
    record = record['character_name']
    with open('audio.wav','wb') as file:
        binary = base64.b64decode(record)
        file.write(binary)
    test = transcribe(file.name)
    test = str(test)
    converted_text = transliterate(test, sanscript.DEVANAGARI,sanscript.ITRANS)
    converted_text = converted_text.lower()
    return jsonify({'hindi':test,
                    'hinglish':converted_text})

if __name__ == '__main__':
    app.run()