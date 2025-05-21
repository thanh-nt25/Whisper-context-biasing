import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.whisper.modeling_whisper import WhisperPreTrainedModel, WhisperModel, shift_tokens_right
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Optional, Tuple, Union, List
from transformers.cache_utils import EncoderDecoderCache

class WhisperForConditionalGenerationWeightCE(WhisperGenerationMixin, WhisperPreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, config, bias_weight=10.0):
        super().__init__(config)
        self.model = WhisperModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.max_target_positions = config.max_target_positions
        self.bias_weight = bias_weight  # Weight for bias word tokens
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def freeze_encoder(self):
        """
        Disable gradient computation for the Whisper encoder.
        """
        self.model.encoder._freeze_parameters()

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask:  Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        bias_spans: Optional[List[List[List[int]]]] = None,  # New parameter for bias spans
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        """
        Args:
            input_features (`torch.FloatTensor`): Input audio features.
            labels (`torch.LongTensor`): Target token IDs for computing loss.
            bias_spans (`List[List[List[int]]]`): Token IDs for bias words per sample.
            ... (other arguments same as WhisperForConditionalGeneration)

        Returns:
            Seq2SeqLMOutput: Model outputs with loss and logits.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if labels.shape[1] > self.max_target_positions:
                raise ValueError(
                    f"Labels' sequence length {labels.shape[1]} cannot exceed the maximum allowed length of {self.max_target_positions} tokens."
                )
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            if bias_spans is not None:
                # Compute weighted cross-entropy loss
                batch_size, seq_len, vocab_size = lm_logits.shape
                weights = torch.ones_like(labels, dtype=torch.float32)  # Default weight of 1.0

                # Assign higher weights to bias word tokens
                for i in range(batch_size):
                    for bias_span in bias_spans[i]:  # Each bias_span is a list of token IDs
                        for token_id in bias_span:
                            mask = labels[i] == token_id
                            weights[i][mask] = self.bias_weight

                # Flatten tensors for loss computation
                logits_flat = lm_logits.view(-1, vocab_size)
                labels_flat = labels.view(-1)
                weights_flat = weights.view(-1)

                # Compute weighted cross-entropy loss
                loss = F.cross_entropy(
                    logits_flat,
                    labels_flat,
                    weight=weights_flat,
                    ignore_index=-100,
                    reduction='mean'
                )
            else:
                # Fall back to standard cross-entropy loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )