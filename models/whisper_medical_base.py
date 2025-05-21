from transformers import WhisperForConditionalGeneration, WhisperProcessor

class WhisperMedicalForConditionalGeneration(WhisperForConditionalGeneration):
    def __init__(self, config, freeze_encoder=True):
        super().__init__(config)

        self.freeze_encoder_flag = freeze_encoder
        if freeze_encoder:
            print("Freezing encoder")
            self._freeze_encoder()
        else:
            print("Load full base model without freezing encoder")

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Frozen encoder")
