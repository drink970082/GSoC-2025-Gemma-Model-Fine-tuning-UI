


class Checkpoint:
    """Class for creating checkpoint instances."""

    @staticmethod
    def create_standard_checkpoint(model_variant: str) -> gm.ckpts.LoadCheckpoint: # type: ignore
        """Create the standard model checkpoint."""
        from gemma import gm
        checkpoint_path = getattr(
            gm.ckpts.CheckpointPath, (model_variant + "_IT").upper()
        )
        return gm.ckpts.LoadCheckpoint(path=checkpoint_path)

    @staticmethod
    def create_lora_checkpoint(model_variant: str) -> gm.ckpts.SkipLoRA: # type: ignore
        """Create the LoRA model checkpoint."""
        from gemma import gm
        base_checkpoint = Checkpoint.create_standard_checkpoint(model_variant)
        return gm.ckpts.SkipLoRA(wrapped=base_checkpoint)
