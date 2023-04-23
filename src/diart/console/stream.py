import argparse
from pathlib import Path

from diart import argdoc
from diart import sources as src
from diart import utils
from diart.inference import StreamingInference
from diart.pipelines import StreamingPipeline, StreamingConfig


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str, help="Path to an audio file | 'microphone' | 'microphone:<DEVICE_ID>'")
    parser.add_argument("--pipeline", default="SpeakerDiarization", type=str,
                        help="Class of the pipeline to optimize. Defaults to 'SpeakerDiarization'")
    parser.add_argument("--whisper", default="small", type=str,
                        help=f"Whisper model for transcription pipeline. Defaults to 'small'")
    parser.add_argument("--language", default="en", type=str,
                        help=f"Transcribe in this language. Defaults to 'en' (English)")
    parser.add_argument("--segmentation", default="pyannote/segmentation", type=str,
                        help=f"{argdoc.SEGMENTATION}. Defaults to pyannote/segmentation")
    parser.add_argument("--embedding", default="pyannote/embedding", type=str,
                        help=f"{argdoc.EMBEDDING}. Defaults to pyannote/embedding")
    parser.add_argument("--step", default=0.5, type=float, help=f"{argdoc.STEP}. Defaults to 0.5")
    parser.add_argument("--latency", default=0.5, type=float, help=f"{argdoc.LATENCY}. Defaults to 0.5")
    parser.add_argument("--tau", default=0.5, type=float, help=f"{argdoc.TAU}. Defaults to 0.5")
    parser.add_argument("--rho", default=0.3, type=float, help=f"{argdoc.RHO}. Defaults to 0.3")
    parser.add_argument("--delta", default=1, type=float, help=f"{argdoc.DELTA}. Defaults to 1")
    parser.add_argument("--gamma", default=3, type=float, help=f"{argdoc.GAMMA}. Defaults to 3")
    parser.add_argument("--beta", default=10, type=float, help=f"{argdoc.BETA}. Defaults to 10")
    parser.add_argument("--max-speakers", default=20, type=int, help=f"{argdoc.MAX_SPEAKERS}. Defaults to 20")
    parser.add_argument("--no-plot", dest="no_plot", action="store_true", help="Skip plotting for faster inference")
    parser.add_argument("--cpu", dest="cpu", action="store_true",
                        help=f"{argdoc.CPU}. Defaults to GPU if available, CPU otherwise")
    parser.add_argument("--output", type=str,
                        help=f"{argdoc.OUTPUT}. Defaults to home directory if SOURCE == 'microphone' or parent directory if SOURCE is a file")
    parser.add_argument("--hf-token", default="true", type=str,
                        help=f"{argdoc.HF_TOKEN}. Defaults to 'true' (required by pyannote)")
    args = parser.parse_args()

    # Resolve pipeline
    pipeline_class = utils.get_pipeline_class(args.pipeline)
    config: StreamingConfig = pipeline_class.get_config_class().from_dict(vars(args))
    pipeline: StreamingPipeline = pipeline_class(config)

    # Manage audio source
    block_size = config.optimal_block_size()
    source_components = args.source.split(":")
    if source_components[0] != "microphone":
        args.source = Path(args.source).expanduser()
        args.output = args.source.parent if args.output is None else Path(args.output)
        padding = config.get_file_padding(args.source)
        audio_source = src.FileAudioSource(args.source, config.sample_rate, padding, block_size)
        pipeline.set_timestamp_shift(-padding[0])
    else:
        args.output = Path("~/").expanduser() if args.output is None else Path(args.output)
        device = int(source_components[1]) if len(source_components) > 1 else None
        audio_source = src.MicrophoneAudioSource(config.sample_rate, block_size, device)

    # Run online inference
    inference = StreamingInference(
        pipeline,
        audio_source,
        batch_size=1,
        do_profile=True,
        show_progress=True,
    )

    # Attach observers for required side effects
    observers = [pipeline.suggest_writer(audio_source.uri, args.output)]
    if not args.no_plot:
        observers.append(pipeline.suggest_display())
    inference.attach_observers(*observers)

    # Run pipeline
    inference()


if __name__ == "__main__":
    run()
