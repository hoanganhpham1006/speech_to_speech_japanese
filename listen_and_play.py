import socket
import numpy as np
import threading
from queue import Queue
from dataclasses import dataclass, field
import sounddevice as sd
from transformers import HfArgumentParser
import wave  # Add this import
import json

@dataclass
class ListenAndPlayArguments:
    send_rate: int = field(
        default=16000,
        metadata={
            "help": "In Hz. Default is 16000."
        }
    )
    recv_rate: int = field(
        default=16000,
        metadata={
            "help": "In Hz. Default is 16000."
        }
    )
    list_play_chunk_size: int = field(
        default=1024,
        metadata={
            "help": "The size of data chunks (in bytes). Default is 1024."
        }
    )
    host: str = field(
        default="localhost",
        metadata={
            "help": "The hostname or IP address for listening and playing. Default is 'localhost'."
        }
    )
    send_port: int = field(
        default=12345,
        metadata={
            "help": "The network port for sending data. Default is 12345."
        }
    )
    recv_port: int = field(
        default=12346,
        metadata={
            "help": "The network port for receiving data. Default is 12346."
        }
    )
    text_port: int = field(
        default=12347,
        metadata={
            "help": "The port for receiving text. Default is 12347."
        }
    )
    emotion_port: int = field(
        default=12348,
        metadata={
            "help": "The port for receiving emotion data. Default is 12348."
        }
    )
    debug: bool = field(
        default=False,
        metadata={
            "help": "If True, sends a sample audio at the beginning"
        }
    )


def listen_and_play(
    send_rate=16000,
    recv_rate=44100,
    list_play_chunk_size=1024,
    host="localhost",
    send_port=12345,
    recv_port=12346,
    text_port=12347,
    emotion_port=12348,
    debug=False,
):
    
    print(f"Listening on {host}:{recv_port} and playing on {host}:{send_port}.")
    print(f"Text will be received on {host}:{text_port}.")
    print(f"Emotion data will be received on {host}:{emotion_port}.")

    send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    send_socket.connect((host, send_port))

    recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    recv_socket.connect((host, recv_port))

    text_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    text_socket.connect((host, text_port))

    emotion_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    emotion_socket.connect((host, emotion_port))

    print("Recording and streaming...")

    stop_event = threading.Event()
    recv_queue = Queue()
    send_queue = Queue()

    def callback_recv(outdata, frames, time, status): 
        if not recv_queue.empty():
            data = recv_queue.get()
            outdata[:len(data)] = data 
            outdata[len(data):] = b'\x00' * (len(outdata) - len(data)) 
        else:
            outdata[:] = b'\x00' * len(outdata) 

    def callback_send(indata, frames, time, status):
        if recv_queue.empty():
            data = bytes(indata)
            send_queue.put(data)

    def send(stop_event, send_queue):
        if debug:
            try:
                # Send audio file once
                with wave.open('kano.wav', 'rb') as wave_file:
                    # Read audio parameters
                    original_rate = wave_file.getframerate()
                    print(f"Original rate: {original_rate}")
                    
                    # Read all frames and convert to numpy array
                    frames = wave_file.readframes(wave_file.getnframes())
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                    
                    # Resample to 16000Hz if needed
                    if original_rate != 16000:
                        samples = len(audio_data)
                        new_samples = int(samples * 16000 / original_rate)
                        audio_data = np.interp(
                            np.linspace(0, samples, new_samples),
                            np.arange(samples),
                            audio_data
                        ).astype(np.int16)
                    
                    # Send resampled data in chunks
                    for i in range(0, len(audio_data), list_play_chunk_size):
                        chunk = audio_data[i:i + list_play_chunk_size].tobytes()
                        send_socket.sendall(chunk)
                
                print("Finished sending debug, switching to microphone...")
            except FileNotFoundError:
                print("Error: debug audio not found in current directory")
            # stop_event.set()
            # return
                
        # Continue with normal microphone streaming
        while not stop_event.is_set():
            data = send_queue.get()
            send_socket.sendall(data)
        
    def recv(stop_event, recv_queue):

        def receive_full_chunk(conn, chunk_size):
            data = b''
            while len(data) < chunk_size:
                packet = conn.recv(chunk_size - len(data))
                if not packet:
                    return None  # Connection has been closed
                data += packet
            return data

        while not stop_event.is_set():
            data = receive_full_chunk(recv_socket, list_play_chunk_size * 2) 
            if data:
                recv_queue.put(data)

    def recv_text(stop_event):
        print("Receiving text...")
        while not stop_event.is_set():
            data = text_socket.recv(1024)
            if not data:
                break
            text_decoded = data.decode('utf-8').strip()
            print(text_decoded)

    def recv_emotion(stop_event):
        print("Receiving emotion data...")
        buffer = ""
        while not stop_event.is_set():
            data = emotion_socket.recv(1024)
            if not data:
                break
            buffer += data.decode('utf-8')
            
            # Process complete JSON messages
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                try:
                    emotion_data = json.loads(line)
                    print(f"Emotion: {emotion_data}")
                except json.JSONDecodeError:
                    print("Error parsing emotion data")

    try: 
        send_stream = sd.RawInputStream(samplerate=send_rate, channels=1, dtype='int16', blocksize=list_play_chunk_size, callback=callback_send)
        recv_stream = sd.RawOutputStream(samplerate=recv_rate, channels=1, dtype='int16', blocksize=list_play_chunk_size, callback=callback_recv)
        threading.Thread(target=send_stream.start).start()
        threading.Thread(target=recv_stream.start).start()

        send_thread = threading.Thread(target=send, args=(stop_event, send_queue))
        send_thread.start()
        recv_thread = threading.Thread(target=recv, args=(stop_event, recv_queue))
        recv_thread.start()

        text_thread = threading.Thread(target=recv_text, args=(stop_event,))
        text_thread.start()

        emotion_thread = threading.Thread(target=recv_emotion, args=(stop_event,))
        emotion_thread.start()

        input("Press Enter to stop...")

    except KeyboardInterrupt:
        print("Finished streaming.")

    finally:
        stop_event.set()
        recv_thread.join()
        send_thread.join()
        text_socket.close()
        send_socket.close()
        recv_socket.close()
        emotion_socket.close()
        print("Connection closed.")


if __name__ == "__main__":
    parser = HfArgumentParser((ListenAndPlayArguments,))
    listen_and_play_kwargs, = parser.parse_args_into_dataclasses()
    listen_and_play(**vars(listen_and_play_kwargs))

