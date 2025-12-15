# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

from modelscope import snapshot_download, HubApi

api = HubApi()
_, cookies = api.login(access_token=os.environ['token'])
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B', cookies=cookies)
snapshot_download('iic/SenseVoiceSmall', local_dir='pretrained_models/SenseVoiceSmall', cookies=cookies)
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd', cookies=cookies)
os.system('cd pretrained_models/CosyVoice-ttsfrd/ && pip install ttsfrd_dependency-0.1-py3-none-any.whl && pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl && apt install -y unzip && rm -rf resource && unzip resource.zip -d .')

from cosyvoice.cli.cosyvoice import AutoModel as CosyVoiceAutoModel
from cosyvoice.utils.file_utils import logging, load_wav
from cosyvoice.utils.common import set_all_random_seed, instruct_list

inference_mode_list = ['3s极速复刻', '自然语言控制']
instruct_dict = {'3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮',
                 '自然语言控制': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入instruct文本\n3. 点击生成音频按钮'}
stream_mode_list = [('否', False)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }

top_db = 60
hop_length = 220
win_length = 440
def postprocess(wav):
    speech = load_wav(wav, target_sr=target_sr, min_sr=16000)
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    torchaudio.save(wav, speech, target_sr)
    return wav


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]

def prompt_wav_recognition(prompt_wav):
    res = asr_model.generate(input=prompt_wav,
                             language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                             use_itn=True,
    )
    text = res[0]["text"].split('|>')[-1]
    return text

def generate_audio(tts_text, mode_checkbox_group, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream):
    stream = False
    if len(tts_text) > 200:
        gr.Warning('您输入的文字过长，请限制在200字以内')
        return (target_sr, default_data)
    sft_dropdown, speed = '', 1.0
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['自然语言控制']:
        if instruct_text == '':
            gr.Warning('您正在使用自然语言控制模式, 请输入instruct文本')
            return (target_sr, default_data)
        if prompt_wav is None:
            gr.Info('您正在使用自然语言控制模式, 请输入prompt音频')
            return (target_sr, default_data)
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
        if prompt_wav is None:
            gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
            return (target_sr, default_data)
        info = torchaudio.info(prompt_wav)
        if info.sample_rate < prompt_sr:
            gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            return (target_sr, default_data)
        if info.num_frames / info.sample_rate > 10:
            gr.Warning('请限制输入音频在10s内，避免推理效果过低')
            return (target_sr, default_data)
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3s极速复刻']:
        if prompt_text == '':
            gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
            return (target_sr, default_data)
        if instruct_text != '':
            gr.Info('您正在使用3s极速复刻模式，instruct文本会被忽略！')
        info = torchaudio.info(prompt_wav)
        if info.num_frames / info.sample_rate > 10:
            gr.Warning('请限制输入音频在10s内，避免推理效果过低')
            return (target_sr, default_data)
    if mode_checkbox_group == '3s极速复刻':
        logging.info('get zero_shot inference request')
        set_all_random_seed(seed)
        speech_list = []
        for i in cosyvoice.inference_zero_shot(tts_text, 'You are a helpful assistant.<|endofprompt|>' + prompt_text, postprocess(prompt_wav), stream=stream, speed=speed):
            speech_list.append(i['tts_speech'])
        return (target_sr, torch.concat(speech_list, dim=1).numpy().flatten())
    elif mode_checkbox_group == '自然语言控制':
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        speech_list = []
        for i in cosyvoice.inference_instruct2(tts_text, instruct_text, postprocess(prompt_wav), stream=stream, speed=speed):
            speech_list.append(i['tts_speech'])
        return (target_sr, torch.concat(speech_list, dim=1).numpy().flatten())
    else:
        gr.Warning('无效的模式选择')


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### 代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    预训练模型 [Fun-CosyVoice3-0.5B-2512](https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) \
                    [CosyVoice2-0.5B](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B) \
                    [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")

        tts_text = gr.Textbox(label="输入合成文本", lines=1, value="Her handwriting is [M][AY0][N][UW1][T]并且很整洁，说明她[h][ào]干净。")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[0])
            instruction_text = gr.Text(label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            stream = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[0][1])
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制prompt音频文件')
        prompt_text = gr.Textbox(label="prompt文本", lines=1, placeholder="请输入prompt文本，支持自动识别，您可以自行修正识别结果...", value='')
        instruct_text = gr.Dropdown(choices=instruct_list, label='选择instruct文本', value=instruct_list[0])

        generate_button = gr.Button("生成音频")

        audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=False)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
        prompt_wav_upload.change(fn=prompt_wav_recognition, inputs=[prompt_wav_upload], outputs=[prompt_text])
        prompt_wav_record.change(fn=prompt_wav_recognition, inputs=[prompt_wav_record], outputs=[prompt_text])
    demo.queue(default_concurrency_limit=4).launch(server_port=50000, server_name='0.0.0.0')


if __name__ == '__main__':
    cosyvoice = CosyVoiceAutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_trt=True, fp16=False)
    sft_spk = cosyvoice.list_available_spks()
    for stream in [False]:
        for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。', 'zero_shot_prompt.wav', stream=stream)):
            continue
    prompt_sr, target_sr = 16000, 24000
    default_data = np.zeros(target_sr)

    model_dir = "pretrained_models/SenseVoiceSmall"
    asr_model = AutoModel(
        model=model_dir,
        disable_update=True,
        log_level='DEBUG',
        device="cuda:0")
    main()