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
    # import speaker_config
    
    question_data = request.get_json()
    print(question_data)

    text = question_data.get('text')
    if not text:
        return app.response_class(
        response=json.dumps({"error_message": "text is required"}),
        status=400,
        mimetype='application/json'
    )
    # speaker = speaker_config.speaker
    # new = speaker_config.new
    
    speaker = question_data.get('speaker')
    if not speaker:
        # 
        speaker = '中文女'

    # start = time.process_time()
    # output = cosyvoice.inference_sft(text,speaker)
    # output = list(output)[0]
    # print(output)
    # end = time.process_time()
    # print("infer time:", end - start)

    # def generate():
    #     buffer = io.BytesIO()
    #     audio_data = output['tts_speech']
    #     torchaudio.save(buffer, audio_data, 22050, format="wav")
    #     buffer.seek(0)
    #     yield buffer.read()
    # with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
    # audio_data = output['tts_speech']
        # torchaudio.save(temp_file.name, audio_data, 22050, format="wav")
    # torchaudio.save('./test.wav', audio_data, 22050, format="wav")
    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
    for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用粤语说这句话', prompt_speech_16k, stream=False)):
        torchaudio.save('./instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
        # def generate():
        #     with open(temp_file.name, 'rb') as f:
        #         data = f.read(8192)
        #         while data:
        #             yield data
        #             data = f.read(8192)
        #     # Clean up the temporary file
        #     os.unlink(temp_file.name)
    # return Response(generate(), mimetype="audio/wav")
    # return Response("Hello, World!", mimetype="text/plain")
    return app.response_class(
        response=json.dumps({"message": "succes"}),
        status=200,
        mimetype='application/json'
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9880)