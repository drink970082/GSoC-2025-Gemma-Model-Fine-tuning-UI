from kauldron import kd


class Loss:
    """Class for creating loss functions."""

    @staticmethod
    def create_loss() -> kd.losses.SoftmaxCrossEntropyWithIntLabels:
        """Create the loss function."""
        return kd.losses.SoftmaxCrossEntropyWithIntLabels(
            logits="preds.logits",
            labels="batch.target",
            mask="batch.loss_mask",
        )
