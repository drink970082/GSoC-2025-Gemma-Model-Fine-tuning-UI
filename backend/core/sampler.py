from typing import Any
from gemma import gm


class Sampler:
    """Factory for creating samplers."""

    @staticmethod
    def create_sampler(
        model: Any, state: Any, tokenizer: Any
    ) -> gm.text.ChatSampler:
        """Create the chat sampler."""
        return gm.text.ChatSampler(
            model=model, params=state.params, tokenizer=tokenizer
        )
