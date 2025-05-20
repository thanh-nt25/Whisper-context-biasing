class WhisperPromptForConditionalGeneration(WhisperPreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model = WhisperPromptModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
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
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.model.encoder._freeze_parameters()

    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_prompts_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,        
        decoder_prompts_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        prompts: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # TODO => prompt에 대한 shift 계산하기
        
        if labels is not None and prompts is None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        elif labels is not None and prompts is not None: # using prompt / training
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.pos_token_id
                )
                

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            # decoder_prompts_ids=decoder_prompts_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            # decoder_prompts_embeds=decoder_prompts_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            if prompts is not None:
                prompt_labels = labels.clone()
                for i, prompt in enumerate(prompts):
                    prompt_labels[i, :len(prompt)] = -100
                prompt_labels = prompt_labels.to(lm_logits.device)
                loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), prompt_labels.reshape(-1))
            else:
                labels = labels.to(lm_logits.device)
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

    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        return_timestamps=None,
        task=None,
        language=None,
        is_multilingual=None,
        prompt_ids: Optional[torch.Tensor] = None,
        num_segment_frames: Optional[int] = None,
        return_token_timestamps: Optional[bool] = None,
        return_segments: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        time_precision: int = 0.02,
        return_dict_in_generate: Optional[bool] = None,
        **kwargs,
    ):
        if "inputs" in kwargs:
            input_features = kwargs.pop("inputs")
            warnings.warn(
                "The input name `inputs` is deprecated. Please make sure to use `input_features` instead.",
                FutureWarning,
            )

        if generation_config is None:
            generation_config = copy.deepcopy(self.generation_config)

        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else generation_config.return_dict_in_generate
        )

        input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
        if num_segment_frames is None:
            num_segment_frames = input_stride * self.config.max_source_positions

        # 1. Check whether we're in shortform or longform mode
        if input_features is not None:
            total_input_frames = input_features.shape[-1]
        elif "encoder_outputs" in kwargs:
            encoder_outputs_shape = (
                kwargs["encoder_outputs"][0].shape
                if isinstance(kwargs["encoder_outputs"], BaseModelOutput)
                else kwargs["encoder_outputs"].shape
            )
            total_input_frames = encoder_outputs_shape[1] * input_stride
        else:
            raise ValueError("Make sure to provide either `input_features` or `encoder_outputs` to `generate`.")

        is_shortform = total_input_frames <= num_segment_frames

        # 2. Make sure the generation config is correctly set depending on whether timestamps are to be returned or not
        if return_timestamps is True:
            if not hasattr(generation_config, "no_timestamps_token_id"):
                raise ValueError(
                    "You are trying to return timestamps, but the generation config is not properly set. "
                    "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`. "
                    "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
                )
            generation_config.return_timestamps = return_timestamps
        elif not is_shortform:
            if return_timestamps is False:
                raise ValueError(
                    "You have passed more than 3000 mel input features (> 30 seconds) which automatically enables long-form generation which "
                    "requires the model to predict timestamp tokens. Please either pass `return_timestamps=True` or make sure to pass no more than 3000 mel input features."
                )

            if not hasattr(generation_config, "no_timestamps_token_id"):
                raise ValueError(
                    "You have passed more than 3000 mel input features (> 30 seconds) which automatically enables long-form generation which "
                    "requires the generation config to have `no_timestamps_token_id` correctly. "
                    "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`. "
                    "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
                    "or make sure to pass no more than 3000 mel input features."
                )

            logger.info("Setting `return_timestamps=True` for long-form generation.")
            generation_config.return_timestamps = True
        else:
            generation_config.return_timestamps = False

        # 3. Make sure to correctly set language-related parameters
        if is_multilingual is not None:
            if not hasattr(generation_config, "is_multilingual"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `is_multilingual` argument "
                    "to `generate`. Please update the generation config as per the instructions "
                    "https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            generation_config.is_multilingual = is_multilingual

        if hasattr(generation_config, "is_multilingual") and not generation_config.is_multilingual:
            if task is not None or language is not None:
                raise ValueError(
                    "Cannot specify `task` or `language` for an English-only model. If the model is intended to be "
                    "multilingual, pass `is_multilingual=True` to generate, or update the generation config."
                )

        if language is not None:
            if not hasattr(generation_config, "lang_to_id"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `language` argument "
                    "to `generate`. Either set the language using the `forced_decoder_ids` in the model config, "
                    "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            language = language.lower()
            generation_config.language = language
        if task is not None:
            if not hasattr(generation_config, "task_to_id"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `task` argument "
                    "to `generate`. Either set the task using the `forced_decoder_ids` in the model config, "
                    "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            generation_config.task = task

        # 4. Add forced decoder ids depending on passed `language`, `task`,`prompt_ids`, `return_token_timestamps` and `return_timestamps`
        forced_decoder_ids = None
        # Legacy code for backward compatibility
        if hasattr(self.config, "forced_decoder_ids") and self.config.forced_decoder_ids is not None:
            forced_decoder_ids = self.config.forced_decoder_ids
        elif (
            hasattr(self.generation_config, "forced_decoder_ids")
            and self.generation_config.forced_decoder_ids is not None
        ):
            forced_decoder_ids = self.generation_config.forced_decoder_ids
        else:
            forced_decoder_ids = kwargs.get("forced_decoder_ids", None)

        if task is not None or language is not None or (forced_decoder_ids is None and prompt_ids is not None):
            forced_decoder_ids = []
            if hasattr(generation_config, "language"):
                if generation_config.language in generation_config.lang_to_id.keys():
                    language_token = generation_config.language
                elif generation_config.language in TO_LANGUAGE_CODE.keys():
                    language_token = f"<|{TO_LANGUAGE_CODE[generation_config.language]}|>"
                elif generation_config.language in TO_LANGUAGE_CODE.values():
                    language_token = f"<|{generation_config.language}|>"
                else:
                    is_language_code = len(generation_config.language) == 2
                    raise ValueError(
                        f"Unsupported language: {generation_config.language}. Language should be one of:"
                        f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                    )
                if language_token not in generation_config.lang_to_id:
                    raise ValueError(
                        f"{language_token} is not supported by this specific model as it is not in the `generation_config.lang_to_id`."
                        "(You should just add it to the generation config)"
                    )
                forced_decoder_ids.append((1, generation_config.lang_to_id[language_token]))
            else:
                forced_decoder_ids.append((1, None))  # automatically detect the language

            if hasattr(generation_config, "task"):
                if generation_config.task in TASK_IDS:
                    forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
                else:
                    raise ValueError(
                        f"The `{generation_config.task}`task is not supported. The task should be one of `{TASK_IDS}`"
                    )
            elif hasattr(generation_config, "task_to_id"):
                forced_decoder_ids.append((2, generation_config.task_to_id["transcribe"]))  # defaults to transcribe
            if hasattr(generation_config, "no_timestamps_token_id") and not generation_config.return_timestamps:
                idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
                forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))

        if forced_decoder_ids is not None:
            generation_config.forced_decoder_ids = forced_decoder_ids

        if prompt_ids is not None:
            if kwargs.get("decoder_start_token_id") is not None:
                raise ValueError(
                    "When specifying `prompt_ids`, you cannot also specify `decoder_start_token_id` as it gets overwritten."
                )
            prompt_ids = prompt_ids.tolist()
            decoder_start_token_id, *text_prompt_ids = prompt_ids
            # Slicing the text prompt ids in a manner consistent with the OpenAI implementation
            # to accomodate context space for the prefix (see https://github.com/openai/whisper/blob/c09a7ae299c4c34c5839a76380ae407e7d785914/whisper/decoding.py#L599)
            text_prompt_ids = text_prompt_ids[-self.config.max_target_positions // 2 - 1 :]
            # Set the decoder_start_token_id to <|startofprev|>
            kwargs.update({"decoder_start_token_id": decoder_start_token_id})

            # If the user passes `max_new_tokens`, increase its number to account for the prompt
            if kwargs.get("max_new_tokens", None) is not None:
                kwargs["max_new_tokens"] += len(text_prompt_ids)
                if kwargs["max_new_tokens"] >= self.config.max_target_positions:
                    raise ValueError(
                        f"The length of the sliced `prompt_ids` is {len(text_prompt_ids)}, and the `max_new_tokens` "
                        f"{kwargs['max_new_tokens'] - len(text_prompt_ids)}. Thus, the combined length of the sliced "
                        f"`prompt_ids` and `max_new_tokens` is: {kwargs['max_new_tokens']}. This exceeds the "
                        f"`max_target_positions` of the Whisper model: {self.config.max_target_positions}. "
                        "You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, "
                        f"so that their combined length is less that {self.config.max_target_positions}."
                    )

            # Reformat the forced_decoder_ids to incorporate the prompt
            non_prompt_forced_decoder_ids = (
                kwargs.pop("forced_decoder_ids", None) or generation_config.forced_decoder_ids
            )
            forced_decoder_ids = [
                *text_prompt_ids,
                generation_config.decoder_start_token_id,
                *[token for _rank, token in non_prompt_forced_decoder_ids],
            ]
            forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_decoder_ids)]
            generation_config.forced_decoder_ids = forced_decoder_ids

        if return_token_timestamps:
            kwargs["output_attentions"] = True
            return_dict_in_generate = True
            kwargs["output_scores"] = True

            if getattr(generation_config, "task", None) == "translate":
                logger.warning("Token-level timestamps may not be reliable for task 'translate'.")
            if not hasattr(generation_config, "alignment_heads"):
                raise ValueError(
                    "Model generation config has no `alignment_heads`, token-level timestamps not available. "
                    "See https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a on how to add this property to the generation config."
                )

            if kwargs.get("num_frames") is not None:
                generation_config.num_frames = kwargs.pop("num_frames")

        if generation_config.return_timestamps is True:
            last_forced_decoder_ids = (
                generation_config.forced_decoder_ids[-1][-1]
                if hasattr(self.config, "forced_decoder_ids") and self.config.forced_decoder_ids
                else None
            )
            if last_forced_decoder_ids == self.generation_config.no_timestamps_token_id:
                # remove no_timestamp to be forcefully generated if we want to return timestamps
                # this is also important to make sure `WhisperTimeStampLogitsProcessor` functions correctly
                forced_decoder_ids = generation_config.forced_decoder_ids[:-1]
                # Make sure that if list is empty we set it to None
                generation_config.forced_decoder_ids = None if len(forced_decoder_ids) == 0 else forced_decoder_ids

            timestamp_processor = [WhisperTimeStampLogitsProcessor(generation_config)]
            logits_processor = (
                timestamp_processor if logits_processor is None else timestamp_processor + logits_processor
            )

        # 5. If we're in shortform mode, simple generate the whole input at once and return the output
        if is_shortform:
            outputs = super().generate(
                input_features,
                generation_config,
                logits_processor,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                synced_gpus,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
            )

            if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
                num_frames = getattr(generation_config, "num_frames", None)
                outputs["token_timestamps"] = self._extract_token_timestamps(
                    outputs, generation_config.alignment_heads, num_frames=num_frames
                )

            return outputs

        # 6. Else we're in longform mode which is more complex. We need to chunk the audio input depending on when the model generated
        # timestamp tokens
        # 6.1 Set running parameters for while loop
        if not return_segments and return_dict_in_generate:
            raise ValueError(
                "Make sure to set `return_segments=True` to return generation outputs as part of the `'segments' key.`"
            )

        # if input is longer than 30 seconds we default to long-form generation
        timestamp_begin = self.generation_config.no_timestamps_token_id + 1
        # input stride is mel frames per encoder output vector which is the product of all conv strides
        batch_size = input_features.shape[0]

        if batch_size > 1 and attention_mask is None:
            raise ValueError(
                "When doing long-form audio transcription, make sure to pass an `attention_mask`. You can retrieve the `attention_mask` by doing `processor(audio, ..., return_attention_mask=True)` "
            )
        elif batch_size > 1:
            max_frames = attention_mask.sum(-1).cpu().to(torch.long)
            seek = torch.zeros((batch_size,), dtype=torch.long)
        else:
            max_frames = torch.ones((1,), dtype=torch.long) * total_input_frames
            seek = torch.zeros((1,), dtype=torch.long)

        current_segments = [[] for _ in range(batch_size)]
        cur_to_prev_index_map = list(range(batch_size))

        # batch size can decrease during the run
        cur_bsz = prev_bsz = batch_size

        # 6.2 Transcribe audio until we reach the end of all input audios
        while (seek < max_frames).any():
            prev_bsz = cur_bsz

            # 6.3 NOTE: When in longform transcription mode and batch size > 1 we need to dynamically reduce the batch size during the loop
            # in case one audio finished earlier than another one. Thus, we need to keep a table of "previous-index-2-current-index" in order
            # to know which original audio is being decoded
            new_cur_to_prev_index_map = []
            for i in range(prev_bsz):
                prev_i = cur_to_prev_index_map[i]
                if seek[prev_i] >= max_frames[prev_i]:
                    cut_index = i + (cur_bsz - prev_bsz)
                    cur_bsz -= 1
                    input_features = torch.cat([input_features[:cut_index], input_features[cut_index + 1 :]], dim=0)
                else:
                    # cut out index that goes away
                    new_cur_to_prev_index_map.append(prev_i)

            # 6.4  Set updated index map, duration of previously decoded chunks and number of max frames of current decoding chunk
            cur_to_prev_index_map = new_cur_to_prev_index_map
            time_offset = seek * time_precision / input_stride
            seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)

            # 6.5 Make sure that all inputs are padded to the same input length
            segment_input = []
            for i in range(cur_bsz):
                prev_i = cur_to_prev_index_map[i]
                segment_input_slice = input_features[
                    i : i + 1, :, seek[prev_i] : seek[prev_i] + seek_num_frames[prev_i]
                ]

                if segment_input_slice.shape[-1] < num_segment_frames:
                    # pad to 3000 if necessary
                    segment_input_slice = F.pad(
                        segment_input_slice, pad=(0, num_segment_frames - segment_input_slice.shape[-1])
                    )

                segment_input.append(segment_input_slice)

            segment_input = torch.cat(segment_input, dim=0)

            # 6.6 Batch generate current chunk
            seek_outputs = super().generate(
                segment_input,
                generation_config,
                logits_processor,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                synced_gpus,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
            )

            if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
                num_frames = getattr(generation_config, "num_frames", None)
                seek_outputs["token_timestamps"] = self._extract_token_timestamps(
                    seek_outputs, generation_config.alignment_heads, num_frames=num_frames
                )

            if return_dict_in_generate:
                seek_sequences = seek_outputs["sequences"]
                seek_outputs = [
                    {k: v[i] for k, v in seek_outputs.items()}
                    for i in range(next(iter(seek_outputs.values())).size(0))
                ]
            else:
                seek_sequences = seek_outputs

            # 6.7 Loop over each decoded audio individually as each decoding can be of a different length
            for i, seek_sequence in enumerate(seek_sequences):
                prev_i = cur_to_prev_index_map[i]

                # make sure we cut a predicted EOS token if we are not finished with the generation yet
                is_not_final = (seek[prev_i] + num_segment_frames) < max_frames[prev_i]
                if is_not_final and seek_sequence[-1] == self.generation_config.eos_token_id:
                    seek_sequence = seek_sequence[:-1]

                # remove all padding tokens
                if seek_sequence[-1] == self.generation_config.pad_token_id:
                    num_paddings = (seek_sequence == self.generation_config.pad_token_id).sum()
                    seek_sequence = seek_sequence[:-num_paddings]

                segments, segment_offset = self._retrieve_segment(
                    seek_sequence=seek_sequence,
                    seek_outputs=seek_outputs,
                    time_offset=time_offset,
                    timestamp_begin=timestamp_begin,
                    seek_num_frames=seek_num_frames,
                    cur_bsz=cur_bsz,
                    time_precision=time_precision,
                    input_stride=input_stride,
                    prev_idx=prev_i,
                    idx=i,
                )

                current_segments[prev_i] += segments
                seek[prev_i] += segment_offset

        # 7. Once all segments are added to the list of all segments, called `current_segments`, we extract the predicted
        # output tokens from the list of dicts. If we use batch size > 1, we make sure to pad the output
        sequences = []
        max_total_length = 0
        for current_segment_list in current_segments:
            sequences.append(torch.cat([d["tokens"] for d in current_segment_list], dim=-1))
            max_total_length = max(max_total_length, len(sequences[-1]))

        for i in range(batch_size):
            sequences[i] = F.pad(
                sequences[i], pad=(0, max_total_length - len(sequences[i])), value=self.generation_config.pad_token_id
            )

        sequences = torch.stack(sequences, dim=0)

        # 8. If we return all segments, the predicted output sequences are put under `"sequences"`.
        if return_segments:
            return {"sequences": sequences, "segments": current_segments}

        return sequences

    @staticmethod
    def _retrieve_segment(
        seek_sequence,
        seek_outputs,
        time_offset,
        timestamp_begin,
        seek_num_frames,
        cur_bsz,
        time_precision,
        input_stride,
        prev_idx,
        idx,
    ):
        # find the predicted "end of segment" predictions of Whisper
        # "end of segment" predictions occur whenever Whisper predicts a timestamp token
        timestamp_tokens: torch.Tensor = seek_sequence.ge(timestamp_begin)
        single_timestamp_ending = timestamp_tokens[-2:].tolist() == cur_bsz * [[False, True]]
        timestamp_segment_indices = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]

        # If whisper predicted a "end of segment" via a timestep token, let's go ever each
        # "end of segment" prediction and slice the decoding into segments accordingly
        if len(timestamp_segment_indices) > 0:
            # if the output contains two consecutive timestamp tokens
            slices = timestamp_segment_indices.tolist()
            segments = []
            if single_timestamp_ending:
                slices.append(len(seek_sequence))

            last_slice = 0
            # Add each segment to list of all segments
            for current_slice in slices:
                sliced_tokens = seek_sequence[last_slice + 1 : current_slice + 1]
                start_timestamp_pos = sliced_tokens[0].item() - timestamp_begin
                end_timestamp_pos = sliced_tokens[-1].item() - timestamp_begin
                segments.append(
                    {
                        "start": time_offset[prev_idx] + start_timestamp_pos * time_precision,
                        "end": time_offset[prev_idx] + end_timestamp_pos * time_precision,
                        "tokens": sliced_tokens,
                        "result": seek_outputs[idx],
                    }
                )
                last_slice = current_slice

            if single_timestamp_ending:
                # single timestamp at the end means no speech after the last timestamp.
                segment_offset = seek_num_frames[prev_idx]
            else:
                # otherwise, ignore the unfinished segment and seek to the last timestamp
                # here we throw away all predictions after the last predicted "end of segment"
                # since we are cutting right in the middle of an audio
                last_timestamp_pos = seek_sequence[last_slice].item() - timestamp_begin
                segment_offset = last_timestamp_pos * input_stride
        else:
            # If whisper does not predict any "end of segment" token, then
            # the whole decoding is considered a segment and we add it to the list of segments
            timestamps = seek_sequence[timestamp_tokens.nonzero().flatten()]
            last_timestamp_pos = seek_num_frames[prev_idx]
            if timestamps.numel() > 0 and timestamps[-1].item() != timestamp_begin:
                # no consecutive timestamps but it has a timestamp; use the last one.
                last_timestamp_pos = timestamps[-1].item() - timestamp_begin

            segments = [
                {
                    "start": time_offset[prev_idx],
                    "end": time_offset[prev_idx] + last_timestamp_pos * time_precision,
                    "tokens": seek_sequence,
                    "result": seek_outputs[idx],
                }
            ]
            segment_offset = seek_num_frames[prev_idx]

        return segments, segment_offset

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": None,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    def _extract_token_timestamps(self, generate_outputs, alignment_heads, time_precision=0.02, num_frames=None):
        """
        Calculates token-level timestamps using the encoder-decoder cross-attentions and dynamic time-warping (DTW) to
        map each output token to a position in the input audio. If `num_frames` is specified, the encoder-decoder
        cross-attentions will be cropped before applying DTW.

        Returns:
            tensor containing the timestamps in seconds for each predicted token
        """
        # Create a list with `decoder_layers` elements, each a tensor of shape
        # (batch size, attention_heads, output length, input length).
        cross_attentions = []
        for i in range(self.config.decoder_layers):
            cross_attentions.append(torch.cat([x[i] for x in generate_outputs.cross_attentions], dim=2))

        # Select specific cross-attention layers and heads. This is a tensor
        # of shape (batch size, num selected, output length, input length).
        weights = torch.stack([cross_attentions[l][:, h] for l, h in alignment_heads])
        weights = weights.permute([1, 0, 2, 3])

        if "beam_indices" in generate_outputs:
            # If beam search has been used, the output sequences may have been generated for more timesteps than their sequence_lengths
            # since the beam search strategy chooses the most probable sequences at the end of the search.
            # In that case, the cross_attentions weights are too long and we have to make sure that they have the right output_length
            weight_length = (generate_outputs.beam_indices != -1).sum(-1).max()
            weights = weights[:, :, :weight_length]

            # If beam index is still -1, it means that the associated token id is EOS
            # We need to replace the index with 0 since index_select gives an error if any of the indexes is -1.
            beam_indices = generate_outputs.beam_indices[:, :weight_length]
            beam_indices = beam_indices.masked_fill(beam_indices == -1, 0)

            # Select the cross attention from the right beam for each output sequences
            weights = torch.stack(
                [
                    torch.index_select(weights[:, :, i, :], dim=0, index=beam_indices[:, i])
                    for i in range(beam_indices.shape[1])
                ],
                dim=2,
            )

        timestamps = torch.zeros_like(generate_outputs.sequences, dtype=torch.float32)
        batch_size = timestamps.shape[0]

        if num_frames is not None:
            # two cases:
            # 1. num_frames is the same for each sample -> compute the DTW matrix for each sample in parallel
            # 2. num_frames is different, compute the DTW matrix for each sample sequentially

            # we're using np.unique because num_frames can be int/list/tuple
            if len(np.unique(num_frames)) == 1:
                # if num_frames is the same, no need to recompute matrix, std and mean for each element of the batch
                num_frames = num_frames if isinstance(num_frames, int) else num_frames[0]

                weights = weights[..., : num_frames // 2]
            else:
                # num_frames is of shape (batch_size,) whereas batch_size is truely batch_size*num_return_sequences
                repeat_time = batch_size if isinstance(num_frames, int) else batch_size // len(num_frames)
                num_frames = np.repeat(num_frames, repeat_time)

        if num_frames is None or isinstance(num_frames, int):
            # Normalize and smoothen the weights.
            std = torch.std(weights, dim=-2, keepdim=True, unbiased=False)
            mean = torch.mean(weights, dim=-2, keepdim=True)
            weights = (weights - mean) / std
            weights = _median_filter(weights, self.config.median_filter_width)

            # Average the different cross-attention heads.
            weights = weights.mean(dim=1)

        # Perform dynamic time warping on each element of the batch.
        for batch_idx in range(batch_size):
            if num_frames is not None and isinstance(num_frames, (tuple, list, np.ndarray)):
                matrix = weights[batch_idx, ..., : num_frames[batch_idx] // 2]

                # Normalize and smoothen the weights.
                std = torch.std(matrix, dim=-2, keepdim=True, unbiased=False)
                mean = torch.mean(matrix, dim=-2, keepdim=True)
                matrix = (matrix - mean) / std
                matrix = _median_filter(matrix, self.config.median_filter_width)

                # Average the different cross-attention heads.
                matrix = matrix.mean(dim=0)
            else:
                matrix = weights[batch_idx]

            text_indices, time_indices = _dynamic_time_warping(-matrix.cpu().double().numpy())
            jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
            jump_times = time_indices[jumps] * time_precision
            timestamps[batch_idx, 1:] = torch.tensor(jump_times)

        return timestamps

