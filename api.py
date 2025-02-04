import os, sys, json, time, io
from flask import Flask, request, Response
from flask_cors import CORS
import torchaudio
import tempfile
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav


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
    print(question_data)

    text = question_data.get('text')
    if not text:
        return app.response_class(
        response=json.dumps({"error_message": "text is required"}),
        status=400,
        mimetype='application/json'
    )
    
    speaker = question_data.get('speaker')
    if not speaker:
        speaker = '中文女'
    # output = cosyvoice.inference_sft(text,speaker)
    # output = list(output)[0]

    # def generate():
    #     buffer = io.BytesIO()
    #     buffer.seek(0)
    #     yield buffer.read()
    start = time.process_time()
    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
    for i, j in enumerate(cosyvoice.inference_instruct2(text, '用粤语说这句话', prompt_speech_16k, stream=False)):
        torchaudio.save('./instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    end = time.process_time()
    print("infer time:", end - start)
        # def generate():
        #     with open(temp_file.name, 'rb') as f:
        #         data = f.read(8192)
        #         while data:
        #             yield data
        #             data = f.read(8192)
        #     # Clean up the temporary file
        #     os.unlink(temp_file.name)
    # return Response(generate(), mimetype="audio/wav")

    return app.response_class(
        response=json.dumps({"message": "success"}),
        status=200,
        mimetype='application/json'
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9880)