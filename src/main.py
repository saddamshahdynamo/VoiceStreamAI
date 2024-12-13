import argparse
import asyncio
import json
import logging

from asr.asr_factory import ASRFactory
from vad.pyannote_vad import PyannoteVAD
from vad.vad_factory import VADFactory
from server import Server


def parse_args():
    parser = argparse.ArgumentParser(
        description="VoiceStreamAI Server: Real-time audio transcription "
        "using self-hosted Whisper and WebSocket."
    )
    parser.add_argument(
        "--vad-type",
        type=str,
        default="pyannote",
        help="Type of VAD pipeline to use (e.g., 'pyannote')",
    )
    parser.add_argument(
        "--vad-args",
        type=str,
        default='{"auth_token": "hf_dYqOwMStIaGrVLIlAOevphYVjCyNGSBOTLs"}',
        help="JSON string of additional arguments for VAD pipeline",
    )
    parser.add_argument(
        "--asr-type",
        type=str,
        default="whisper",
        help="Type of ASR pipeline to use (e.g., 'whisper')",
    )
    parser.add_argument(
        "--asr-args",
        type=str,
        default='{"model_size": "base", "model_name": "tarteel-ai/whisper-base-ar-quran"}',
        help="JSON string of additional arguments for ASR pipeline",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the WebSocket server",
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="Port for the WebSocket server"
    )
    parser.add_argument(
        "--certfile",
        type=str,
        default=None,
        help="The path to the SSL certificate (cert file) if using secure "
        "websockets",
    )
    parser.add_argument(
        "--keyfile",
        type=str,
        default=None,
        help="The path to the SSL key file if using secure websockets",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="error",
        choices=["debug", "info", "warning", "error"],
        help="Logging level: debug, info, warning, error. default: error",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(args.log_level.upper())

    try:
        # vad_args = json.loads(args.vad_args)
        vad_args = json.loads('{"auth_token": "hf_dYqOwMStIaGrVLIlAOevphYVjCyNGSBOTLs"}')
        asr_args = json.loads(args.asr_args)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON arguments: {e}")
        return

    vad_pipeline = VADFactory.create_vad_pipeline(args.vad_type, **vad_args)
    asr_pipeline = ASRFactory.create_asr_pipeline(args.asr_type, **asr_args)

    server = Server(
        vad_pipeline,
        asr_pipeline,
        host=args.host,
        port=args.port,
        sampling_rate=16000,
        samples_width=2,
        certfile=args.certfile,
        keyfile=args.keyfile,
    )

    asyncio.run(server.start())


if __name__ == "__main__":
    main()
