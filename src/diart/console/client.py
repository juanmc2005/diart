import argparse
from pathlib import Path
from threading import Thread
from typing import Text, Optional

import rx.operators as ops
from websocket import WebSocket

from diart import argdoc
from diart import sources as src
from diart import utils


def send_audio(ws: WebSocket, source: Text, step: float, sample_rate: int):
    # Create audio source
    source_components = source.split(":")
    if source_components[0] != "microphone":
        audio_source = src.FileAudioSource(source, sample_rate, block_duration=step)
    else:
        device = int(source_components[1]) if len(source_components) > 1 else None
        audio_source = src.MicrophoneAudioSource(step, device)

    # Encode audio, then send through websocket
    audio_source.stream.pipe(ops.map(utils.encode_audio)).subscribe_(ws.send)

    # Start reading audio
    audio_source.read()


def receive_audio(ws: WebSocket, output: Optional[Path]):
    while True:
        message = ws.recv()
        print(f"Received: {message}", end="")
        if output is not None:
            with open(output, "a") as file:
                file.write(message)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=str,
        help="Path to an audio file | 'microphone' | 'microphone:<DEVICE_ID>'",
    )
    parser.add_argument("--host", required=True, type=str, help="Server host")
    parser.add_argument("--port", required=True, type=int, help="Server port")
    parser.add_argument(
        "--step", default=0.5, type=float, help=f"{argdoc.STEP}. Defaults to 0.5"
    )
    parser.add_argument(
        "-sr",
        "--sample-rate",
        default=16000,
        type=int,
        help=f"{argdoc.SAMPLE_RATE}. Defaults to 16000",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        help="Output RTTM file. Defaults to no writing",
    )
    args = parser.parse_args()

    # Run websocket client
    ws = WebSocket()
    ws.connect(f"ws://{args.host}:{args.port}")
    sender = Thread(
        target=send_audio, args=[ws, args.source, args.step, args.sample_rate]
    )
    receiver = Thread(target=receive_audio, args=[ws, args.output_file])
    sender.start()
    receiver.start()


if __name__ == "__main__":
    run()
