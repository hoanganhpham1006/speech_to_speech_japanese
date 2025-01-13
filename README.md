<div align="center">
  <div>&nbsp;</div>
  <img src="logo.png" width="600"/> 
</div>

# Speech To Speech: an effort for an open-sourced and modular GPT4-o

fork from: [https://github.com/eustlb/speech-to-speech](https://github.com/eustlb/speech-to-speech)

# Japanese support

Python 3.10で動作確認済み

```bash
git clone https://github.com/shi3z/speech-to-speech-japanese.git
cd speech-to-speech-japanese
pip install git+https://github.com/nltk/nltk.git@3.8.2
git clone https://github.com/reazon-research/ReazonSpeech
pip install Cython
pip install ReazonSpeech/pkg/nemo-asr
git clone https://github.com/myshell-ai/MeloTTS
cd MeloTTS
pip install -e .
python -m unidic download
cd ..
pip install -r requirements.txt
pip install transformers==4.44.1
pip install mlx-lm
pip install protobuf --upgrade
python s2s_pipeline.py --mode local --device mps
```
MacBookPro M2 Max(32GB)で動作確認済
MacBook M1(16GB)でも動作確認済み( @necobit (https://github.com/necobit) )

## 📖 Quick Index
* [Approach](#approach)
  - [Structure](#structure)
  - [Modularity](#modularity)
* [Setup](#setup)
* [Usage](#usage)
  - [Server/Client approach](#serverclient-approach)
  - [Local approach](#local-approach)
* [Command-line usage](#command-line-usage)
  - [Model parameters](#model-parameters)
  - [Generation parameters](#generation-parameters)
  - [Notable parameters](#notable-parameters)

## Approach

### Structure
This repository implements a speech-to-speech cascaded pipeline with consecutive parts:
1. **Voice Activity Detection (VAD)**: [silero VAD v5](https://github.com/snakers4/silero-vad)
2. **Speech to Text (STT)**: Whisper checkpoints (including [distilled versions](https://huggingface.co/distil-whisper))
3. **Language Model (LM)**: Any instruct model available on the [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending)! 🤗
4. **Text to Speech (TTS)**: [Parler-TTS](https://github.com/huggingface/parler-tts)🤗

### Modularity
The pipeline aims to provide a fully open and modular approach, leveraging models available on the Transformers library via the Hugging Face hub. The level of modularity intended for each part is as follows:
- **VAD**: Uses the implementation from [Silero's repo](https://github.com/snakers4/silero-vad).
- **STT**: Uses Whisper models exclusively; however, any Whisper checkpoint can be used, enabling options like [Distil-Whisper](https://huggingface.co/distil-whisper/distil-large-v3) and [French Distil-Whisper](https://huggingface.co/eustlb/distil-large-v3-fr).
- **LM**: This part is fully modular and can be changed by simply modifying the Hugging Face hub model ID. Users need to select an instruct model since the usage here involves interacting with it.
- **TTS**: The mini architecture of Parler-TTS is standard, but different checkpoints, including fine-tuned multilingual checkpoints, can be used.

The code is designed to facilitate easy modification. Each component is implemented as a class and can be re-implemented to match specific needs.

## Setup

Clone the repository:
```bash
git clone https://github.com/eustlb/speech-to-speech.git
cd speech-to-speech
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The pipeline can be run in two ways:
- **Server/Client approach**: Models run on a server, and audio input/output are streamed from a client.
- **Local approach**: Uses the same client/server method but with the loopback address.

### Server/Client Approach

To run the pipeline on the server:
```bash
python s2s_pipeline.py --recv_host 0.0.0.0 --send_host 0.0.0.0 --recv_port 8005 --send_port 8006 --text_port 8007 --text_host 0.0.0.0 --stt_compile_mode reduce-overhead
```

Then run the client locally to handle sending microphone input and receiving generated audio:
```bash
python listen_and_play.py --host <IP address of your server> --send_port 8005 --recv_port 8006 --text_port 8007 --debug
```

### Local Approach
Simply use the loopback address:
```bash
python s2s_pipeline.py --recv_host localhost --send_host localhost
python listen_and_play.py --host localhost
```