import os
import urllib.request
from app import app
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import whisper
import json
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

ALLOWED_EXTENSIONS = set(['wav'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/getfile', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(os.getcwd(), filename)
        file.save(filepath)
        hindi = transcribe(filepath)
        converted_text = transliterate(hindi, sanscript.DEVANAGARI,sanscript.ITRANS).lower()
        english = translator.translate(hindi).text
        resp = jsonify({"Hindi": hindi,"Hinglish": converted_text,"English":english})
        resp.status_code = 200
        return resp
    else:
        resp = jsonify({'message' : 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
        resp.status_code = 400
        return resp


if __name__ == "__main__":
    app.run(host = '0.0.0.0',port = 8000)