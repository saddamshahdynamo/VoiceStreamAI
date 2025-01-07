import os
from typing import ClassVar


class FilePaths:
    ROOT_DIR: ClassVar[str] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    AUDIO_FILES_DIR: ClassVar[str] = os.path.join(ROOT_DIR, "audio_files")
    PARTIAL_AUDIO_DIR: ClassVar[str] = os.path.join(ROOT_DIR, "partial_audio_chunks")
    NOISE_DIR: ClassVar[str] = os.path.join(ROOT_DIR, "noise")
    AUDIO_TRANSCRIPTION_DIR: ClassVar[str] = os.path.join(ROOT_DIR, "audio_transcription")
    PARTIAL_AUDIO_TRANSCRIPTION_DIR: ClassVar[str] = os.path.join(ROOT_DIR, "audio_transcription_partial")
    LOG_DIR: ClassVar[str] = os.path.join(ROOT_DIR, "log_dir")
    NOISE_LOG_FILE: ClassVar[str] = os.path.join(NOISE_DIR, "noise_entries.log")

    def __new__(cls) -> None:
        raise TypeError(f'{cls.__name__} class is static and cannot be instantiated')

    @classmethod
    def create_directories(cls) -> str:
        # Create directories if they don't exist
        os.makedirs(cls.AUDIO_FILES_DIR, exist_ok=True)
        os.makedirs(cls.PARTIAL_AUDIO_DIR, exist_ok=True)
        os.makedirs(cls.PARTIAL_AUDIO_DIR, exist_ok=True)
        os.makedirs(cls.NOISE_DIR, exist_ok=True)
        os.makedirs(cls.AUDIO_TRANSCRIPTION_DIR, exist_ok=True)
        os.makedirs(cls.PARTIAL_AUDIO_TRANSCRIPTION_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        return "Directories created successfully"


# Initialize directories when module is loaded
FilePaths.create_directories()