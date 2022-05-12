import argparse

import torch

import diart.argdoc as argdoc
from diart.inference import Benchmark
from diart.pipelines import OnlineSpeakerDiarization, PipelineConfig

if __name__ == "__main__":
    # Define script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="Directory with audio files CONVERSATION.(wav|flac|m4a|...)")
    parser.add_argument("--reference", type=str, help="Optional. Directory with RTTM files CONVERSATION.rttm. Names must match audio files")
    parser.add_argument("--step", default=0.5, type=float, help=f"{argdoc.STEP}. Defaults to 0.5")
    parser.add_argument("--latency", default=0.5, type=float, help=f"{argdoc.LATENCY}. Defaults to 0.5")
    parser.add_argument("--tau", default=0.5, type=float, help=f"{argdoc.TAU}. Defaults to 0.5")
    parser.add_argument("--rho", default=0.3, type=float, help=f"{argdoc.RHO}. Defaults to 0.3")
    parser.add_argument("--delta", default=1, type=float, help=f"{argdoc.DELTA}. Defaults to 1")
    parser.add_argument("--gamma", default=3, type=float, help=f"{argdoc.GAMMA}. Defaults to 3")
    parser.add_argument("--beta", default=10, type=float, help=f"{argdoc.BETA}. Defaults to 10")
    parser.add_argument("--max-speakers", default=20, type=int, help=f"{argdoc.MAX_SPEAKERS}. Defaults to 20")
    parser.add_argument("--batch-size", default=32, type=int, help="For segmentation and embedding pre-calculation. If BATCH_SIZE < 2, run fully online and estimate real-time latency. Defaults to 32")
    parser.add_argument("--gpu", dest="gpu", action="store_true", help=argdoc.GPU)
    parser.add_argument("--output", type=str, help=f"{argdoc.OUTPUT}. Defaults to `root`")
    args = parser.parse_args()

    # Set benchmark configuration
    benchmark = Benchmark(args.root, args.reference, args.output)

    # Define online speaker diarization pipeline
    pipeline = OnlineSpeakerDiarization(PipelineConfig(
        step=args.step,
        latency=args.latency,
        tau_active=args.tau,
        rho_update=args.rho,
        delta_new=args.delta,
        gamma=args.gamma,
        beta=args.beta,
        max_speakers=args.max_speakers,
        device=torch.device("cuda") if args.gpu else None,
    ))

    benchmark(pipeline, args.batch_size)
