import argparse
from pathlib import Path
from threading import Event, Thread
from typing import Optional, Text

import rx.operators as ops
from websocket import WebSocket, WebSocketException

from diart import argdoc
from diart import sources as src
from diart import utils


def send_audio(
    ws: WebSocket, source: Text, step: float, sample_rate: int, stop_event: Event
):
    try:
        # Create audio source
        source_components = source.split(":")
        if source_components[0] != "microphone":
            audio_source = src.FileAudioSource(source, sample_rate, block_duration=step)
        else:
            device = int(source_components[1]) if len(source_components) > 1 else None
            audio_source = src.MicrophoneAudioSource(step, device)

        # Encode audio, then send through websocket
        def on_next(data):
            if not stop_event.is_set():
                try:
                    ws.send(utils.encode_audio(data))
                except WebSocketException:
                    stop_event.set()

        audio_source.stream.subscribe_(on_next)

        # Start reading audio
        audio_source.read()
    except Exception as e:
        print(f"Error in send_audio: {e}")
        stop_event.set()


def receive_audio(ws: WebSocket, output: Optional[Path], stop_event: Event):
    try:
        while not stop_event.is_set():
            try:
                message = ws.recv()
                print(f"Received: {message}", end="")
                if output is not None:
                    with open(output, "a") as file:
                        file.write(message)
            except WebSocketException:
                break
    except Exception as e:
        print(f"Error in receive_audio: {e}")
    finally:
        stop_event.set()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=str,
        help="Path to an audio file | 'microphone' | 'microphone:<DEVICE_ID>'",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", type=str, help="Server host. Defaults to 0.0.0.0"
    )
    parser.add_argument(
        "--port", default=7007, type=int, help="Server port. Defaults to 7007"
    )
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
    stop_event = Event()

    try:
        ws.connect(f"ws://{args.host}:{args.port}")

        # Wait for READY signal from server
        print("Waiting for server to be ready...", end="", flush=True)
        while not stop_event.is_set():
            try:
                message = ws.recv()
                if message.strip() == "READY":
                    print(" OK")
                    break
                print(f"\nUnexpected message while waiting for READY: {message}")
            except WebSocketException as e:
                print(f"\nWebSocket error while waiting for server: {e}")
                return

        # Start threads for sending and receiving audio
        sender = Thread(
            target=send_audio,
            args=[ws, args.source, args.step, args.sample_rate, stop_event],
        )
        receiver = Thread(target=receive_audio, args=[ws, args.output_file, stop_event])

        sender.start()
        receiver.start()

        try:
            # Wait for threads to complete or for keyboard interrupt
            sender.join()
            receiver.join()
        except KeyboardInterrupt:
            print("\nShutting down...")
            stop_event.set()

    except Exception as e:
        print(f"Error: {e}")

    finally:
        stop_event.set()
        try:
            ws.close()
        except WebSocketException:
            print("Error closing WebSocket")
        except Exception as e:
            print(f"Unexpected error closing WebSocket: {e}")


if __name__ == "__main__":
    run()
