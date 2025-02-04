import os, sys, json
from flask import Flask, request
from flask_cors import CORS
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import numpy as np
import simpleaudio as sa


app = Flask(__name__)
CORS(app, cors_allowed_origins="*")
CORS(app, supports_credentials=True)

# set environemnt variable
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/third_party/AcademiCodec')
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')
os.environ['PYTHONPATH'] = f'{ROOT_DIR}/third_party/AcademiCodec:{ROOT_DIR}/third_party/Matcha-TTS'

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')

@app.route("/speakers", methods=['GET'])
def speakers():
    voices = []
    speakers_list = cosyvoice.list_available_spks()

    for speaker in speakers_list:
        voices.append({"name":speaker,"voice_id":speaker})

    response = app.response_class(
        response=json.dumps(voices),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route("/tts_to_audio/", methods=['POST'])
def tts_to_audio():
    question_data = request.get_json()
    text = question_data.get('text')
    if not text:
        return app.response_class(
            response=json.dumps({"error_message": "text is required"}),
            status=400,
            mimetype='application/json'
        )
    
    speaker = question_data.get('speaker', '粤语女')
    print(f"speaker: {speaker}")
    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
    for chunk in cosyvoice.inference_instruct2(text, '用粤语说这句话', prompt_speech_16k, stream=False):
    # for chunk in cosyvoice.inference_sft(text, speaker):
        # Convert to int16 format
        audio_data = chunk['tts_speech'].numpy()
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Play audio
        play_obj = sa.play_buffer(audio_data, 1, 2, cosyvoice.sample_rate)
        play_obj.wait_done()
    
    return app.response_class(
        response=json.dumps({"message": "Audio played successfully"}),
        status=200,
        mimetype='application/json'
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9880)