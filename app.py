from flask import Flask,request,jsonify
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import whisper
import json
import base64
from googletrans import Translator

model = whisper.load_model("small")
translator = Translator()

def transcribe(audio):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(fp16 = False,language="hi")
    result = whisper.decode(model, mel, options)
    return result.text

app = Flask(__name__)

@app.route('/',methods = ["POST"])
def welcome():
    record= json.loads(request.data)
    record = record['voice']
    with open('audio.wav','wb') as file:
        binary = base64.b64decode(record)
        file.write(binary)
    test = transcribe('audio.wav')
    test = str(test)
    converted_text = transliterate(test, sanscript.DEVANAGARI,sanscript.ITRANS).lower()
    english = translator.translate(test).text
    return jsonify({"Hindi": test,"Hinglish": converted_text,"English":english})

if __name__ == '_main_':
    app.run(host = '0.0.0.0',port = 8000)