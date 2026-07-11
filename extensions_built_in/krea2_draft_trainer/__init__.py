from toolkit.extension import Extension


class Krea2DraftTrainerExtension(Extension):
    """DRaFT-K reward training stage for Krea 2 (FedorAiToolkit).

    Resumes an SFT-trained LoRA / LoKr and optimizes it directly on
    differentiable face + body similarity rewards computed on images the
    model generates during training.
    """

    uid = "krea2_draft_trainer"
    name = "Krea 2 DRaFT Reward Trainer"

    @classmethod
    def get_process(cls):
        from .Krea2DraftTrainer import Krea2DraftTrainer

        return Krea2DraftTrainer


AI_TOOLKIT_EXTENSIONS = [
    Krea2DraftTrainerExtension,
]
