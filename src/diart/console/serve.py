import argparse
from pathlib import Path

from diart import argdoc
from diart import sources as src
from diart import utils
from diart.inference import StreamingInference
from diart.pipelines import Pipeline


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", type=str, help="Server host")
    parser.add_argument("--port", default=7007, type=int, help="Server port")
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
    parser.add_argument("--duration", type=float,
                        help=f"Duration of the sliding window (in seconds). Default value depends on the pipeline")
    parser.add_argument("--step", default=0.5, type=float, help=f"{argdoc.STEP}. Defaults to 0.5")
    parser.add_argument("--latency", default=0.5, type=float, help=f"{argdoc.LATENCY}. Defaults to 0.5")
    parser.add_argument("--tau", default=0.5, type=float, help=f"{argdoc.TAU}. Defaults to 0.5")
    parser.add_argument("--rho", default=0.3, type=float, help=f"{argdoc.RHO}. Defaults to 0.3")
    parser.add_argument("--delta", default=1, type=float, help=f"{argdoc.DELTA}. Defaults to 1")
    parser.add_argument("--gamma", default=3, type=float, help=f"{argdoc.GAMMA}. Defaults to 3")
    parser.add_argument("--beta", default=10, type=float, help=f"{argdoc.BETA}. Defaults to 10")
    parser.add_argument("--max-speakers", default=20, type=int, help=f"{argdoc.MAX_SPEAKERS}. Defaults to 20")
    parser.add_argument("--cpu", dest="cpu", action="store_true",
                        help=f"{argdoc.CPU}. Defaults to GPU if available, CPU otherwise")
    parser.add_argument("--output", type=Path, help=f"{argdoc.OUTPUT}. Defaults to no writing")
    parser.add_argument("--hf-token", default="true", type=str,
                        help=f"{argdoc.HF_TOKEN}. Defaults to 'true' (required by pyannote)")
    args = parser.parse_args()

    # Resolve pipeline
    pipeline_class = utils.get_pipeline_class(args.pipeline)
    config = pipeline_class.get_config_class().from_dict(vars(args))
    pipeline: Pipeline = pipeline_class(config)

    # Create websocket audio source
    audio_source = src.WebSocketAudioSource(config.sample_rate, args.host, args.port)

    # Run online inference
    inference = StreamingInference(
        pipeline,
        audio_source,
        batch_size=1,
        do_profile=False,
        show_progress=True,
    )

    # Write to disk if required
    if args.output is not None:
        inference.attach_observers(pipeline.suggest_writer(audio_source.uri, args.output))

    # Send back responses as text
    inference.attach_hooks(lambda pred_wav: audio_source.send(utils.serialize_prediction(pred_wav[0])))

    # Run server and pipeline
    inference()


if __name__ == "__main__":
    run()
