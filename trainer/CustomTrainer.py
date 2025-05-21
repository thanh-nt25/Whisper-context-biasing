from transformers import Seq2SeqTrainer

class CustomTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        # Tách bias_spans ra để không truyền vào generate()
        bias_spans = inputs.pop("bias_spans", None)

        # Lưu lại để dùng trong compute_metrics nếu cần
        if bias_spans is not None:
            if not hasattr(self, "_all_bias_spans"):
                self._all_bias_spans = []
            self._all_bias_spans.extend(bias_spans)

        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        if hasattr(self, "_all_bias_spans"):
            output.inputs = {"bias_spans": self._all_bias_spans}
        return output
