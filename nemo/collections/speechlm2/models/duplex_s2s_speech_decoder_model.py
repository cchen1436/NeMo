# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torchaudio
import torch.distributed as dist
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from torch import Tensor
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from transformers import DynamicCache, WhisperFeatureExtractor
import uuid
import os
from nemo.collections.audio.parts.utils.resampling import resample
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.models.duplex_s2s_model import replace_control_speech_codes, tokens_to_str
from nemo.collections.speechlm2.modules import TransformerARSpeechDecoder
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.mos import MOS
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.modules.speech_tokenizer.utils import extract_speech_token
from nemo.collections.speechlm2.modules.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_hf, setup_audio_codec, setup_speech_encoder
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging
import torch.distributed as dist
from nemo.collections.speechlm2.modules.flow_inference import AudioDecoder
import torch.nn.utils.rnn as rnn_utils
from librosa.filters import mel as librosa_mel_fn

def mel_spectrogram(y, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=8000, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    mel_basis = {}
    hann_window = {} # pylint: disable=global-statement,global-variable-not-assigned
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)

    spec = torch.log(torch.clamp(spec, min=1e-5) * 1) # 1e-5 is clip_val

    return spec


class DuplexS2SSpeechDecoderModel(LightningModule, HFHubMixin):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to DuplexS2SModel as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)

        # setup_audio_codec(self)
        self._codebook_size = 16384
        self._num_codebooks = 1


        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        # llm = load_pretrained_hf(self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights).train()
        # self.llm = llm.model  # fetch PretrainedBaseModel from model "ForCausalLM"
        # self.lm_head = llm.lm_head
        # self.embed_tokens = self.llm.embed_tokens
        # del self.llm.embed_tokens
        maybe_install_lora(self)

        # setup_speech_encoder(self)

        # self.speech_generation = TransformerARSpeechDecoder(
        #     speech_decoder_parms=OmegaConf.to_container(self.cfg.speech_decoder),
        #     lantent_dim=self.llm.config.hidden_size,
        #     num_audio_codebooks=self._num_codebooks,
        #     num_audio_tokens_per_codebook=self.speech_vocab_size,
        # )

        self.whispervq = WhisperVQEncoder.from_pretrained(
            "THUDM/glm-4-voice-tokenizer",
            cache_dir='/hfcache',
        ).float().eval()

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            "THUDM/glm-4-voice-tokenizer",
            cache_dir='/hfcache',
        )


        flow_config = os.path.join(self.cfg.pretrained_flow,
                                   "config.yaml")
        flow_checkpoint = os.path.join(
            self.cfg.pretrained_flow,
            'flow.pt')
        hift_checkpoint = os.path.join(
            self.cfg.pretrained_flow,
            'hift.pt')
        self.audio_decoder = AudioDecoder(config_path=flow_config, flow_ckpt_path=flow_checkpoint,
                                     hift_ckpt_path=hift_checkpoint)

        # self.embed_audio_tokens = torch.nn.Embedding(16384, self.llm.config.hidden_size)


        # cached for quicker audio decoding
        self.register_buffer(
            "_control_codes",
            torch.tensor([self.speech_bos_id, self.speech_eos_id, self.speech_delay_id], device=self.device),
        )

        self._use_fsdp = False
        self._use_tp = False


    @property
    def speech_vocab_size(self):
        """Return the size of the audio codec codebook including extra speech BOS and EOS tokens."""
        return self._codebook_size + 4

    @property
    def speech_bos_id(self) -> int:
        """Indicates start of utterance generation (not start of inference!)."""
        return self._codebook_size

    @property
    def speech_eos_id(self) -> int:
        """Indicates end of utterance generation."""
        return self._codebook_size + 1

    @property
    def speech_delay_id(self) -> int:
        """Indicates start of inference (the very first frame)."""
        return self._codebook_size + 2

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        """
        Text pad ID is used as a 'blank' for frames when the model is not speaking
        and for frames where the model is speaking but has already predicted the
        entire text channel's content.

        Example:

            flow:         |---user---||-------assistant--------||-user-|
            text channel:  0000000000  1xxxxxxx0000000000000002  000000

        Where 0 indicates PAD ID, 1 indicates BOS ID, 2 indacates EOS ID,
        and x indicates tokens corresponding to actual text

        """
        return get_pad_id(self.tokenizer)

    def forward(self, input_embeds: Tensor, cache=None, input_audio_tokens=None, text_label=None, loss_mask=None) -> dict[str, Tensor]:
        """
        Separated text and speech prediction:
            - Speech prediction is achieved by a independent AR decoder based on last_hidden_state + audio tokens
            - For KV-cache:
                (1) llm cache depends on input cache is None or Not
                (2) speech_generation cache relys on reset_input_and_kv_cache function.
        """

        out = self.llm(
            inputs_embeds=input_embeds, past_key_values=cache, use_cache=cache is not None, return_dict=True
        )
        B, T = input_embeds.shape[:2]
        text_logits = self.lm_head(out['last_hidden_state'])  # (B, T, text_vocab_size)

        if loss_mask is not None:
            # This is training Mode
            loss_mask = loss_mask[:, :, -1].reshape(loss_mask.size(0), loss_mask.size(1))
            self.speech_generation.reset_input_and_kv_cache(use_cache=False)

        if text_label is not None:
            text_emb = self.embed_tokens(text_label)
        else:
            text_emb = self.embed_tokens(text_logits[:, -1].argmax(dim=-1).unsqueeze(dim=1))

        speech_input_hidden = text_emb * self.cfg.get("word_emb_weight", 1.0) + out['last_hidden_state']

        _, audio_logits = self.speech_generation(
            speech_input_hidden.transpose(0, 1), loss_mask, input_audio_tokens=input_audio_tokens
        )

        audio_logits = audio_logits.view(B, T, self._num_codebooks, self.speech_vocab_size)

        ans = {
            "text_logits": text_logits,
            "audio_logits": audio_logits,
        }
        if cache is not None:
            ans["cache"] = out["past_key_values"]
        return ans

    def prepare_inputs(self, batch: dict):


        target_audio_list = [sample[sample != 0].unsqueeze(0) for sample in batch["target_audio"]]

        target_mel_list = [mel_spectrogram(sample) for sample in target_audio_list]

        speech_feat_lens = torch.tensor([x.shape[2] for x in target_mel_list], device=target_mel_list[0].device)

        speech_feat = torch.nn.utils.rnn.pad_sequence(
            [x.squeeze(0).permute(1, 0) for x in target_mel_list],
            batch_first=True
        ).permute(0, 2, 1)


        with fp32_precision():
            target_audio_list = [resample(target_audio_list[i], 22050, 16000) for i in range(len(target_audio_list))]

            speech_tokens = extract_speech_token(
            self.whispervq,
            self.feature_extractor,
            [(target_audio_list[i], 16000) for i in range(len(target_audio_list))],
            )

        speech_token_len = torch.tensor([len(seq) for seq in speech_tokens], dtype=torch.long, device=self.device)
        max_len = speech_token_len.max().item()
        padded_tensor = torch.zeros(len(speech_tokens), max_len, dtype=torch.long, device=self.device)
        for i, seq in enumerate(speech_tokens):
            padded_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)

        speech_tokens = padded_tensor

        return {
            "speech_feat": speech_feat,
            "speech_feat_len": speech_feat_lens,
            "speech_token": speech_tokens,
            "speech_token_len": speech_token_len,
            "embedding": torch.zeros(speech_tokens.size(0), 192).to(self.device)
        }





        """Prepares input tensors for the model."""
        source_encoded, source_encoded_lens = self.perception(
            input_signal=batch["source_audio"], input_signal_length=batch["source_audio_lens"]
        )


        #
        # speech_feat = mel_spectrogram(batch["target_audio"]) # 1
        # speech_feat_lens = (batch["target_audio_lens"] - 256) // 256 + 1  # 2
        # 【batch, 80, n_feat】

        target_tokens = batch["target_tokens"]

        with fp32_precision():  # resample is fragile to bfloat16 default dtype
            resampled_audio = resample(batch["target_audio"], 22050, 16000)

            target_speech_tokens = extract_speech_token(
                self.whispervq,
                self.feature_extractor,
                [(resampled_audio[i].unsqueeze(0), 16000) for i in range(batch["target_audio"].shape[0])],
            )
        # import pdb; pdb.set_trace()
        target_speech_tokens = torch.tensor(target_speech_tokens, dtype=torch.long, device=self.device)
        # speech_tokens = speech_tokens[:, :max(speech_tokens_lens)] # 4



        min_len = min(source_encoded.shape[1], target_speech_tokens.shape[1], target_tokens.shape[1])
        source_encoded = source_encoded[:, :min_len]
        target_speech_tokens = target_speech_tokens[:, :min_len].unsqueeze(dim=-1)
        target_tokens = target_tokens[:, :min_len]
        source_encoded_lens = torch.clamp_(source_encoded_lens, max=min_len)



        btt = target_tokens[..., None]

        target_speech_tokens = torch.where(btt == self.text_bos_id, self.speech_bos_id, target_speech_tokens)
        target_speech_tokens = torch.where(btt == self.text_eos_id, self.speech_eos_id, target_speech_tokens)

        target_speech_tokens = torch.cat(
            [
                torch.full(
                    [target_speech_tokens.shape[0], 1, target_speech_tokens.shape[-1]],
                    fill_value=self.speech_delay_id,
                    device=self.device,
                    dtype=torch.long,
                ),
                target_speech_tokens[:, :-1],
            ],
            dim=1,
        )

        input_ids = torch.cat([target_speech_tokens, target_tokens[..., None]], dim=-1)

        text_inputs = input_ids[:, :-1, -1]
        text_labels = input_ids[:, 1:, -1]
        audio_inputs = input_ids[:, :-1, :1]
        audio_labels = input_ids[:, 1:, :1]

        input_embeds = self.embed_tokens(text_inputs)


        user_stream = source_encoded[:, :-1] * self.cfg.get("duplex_user_emb_weight", 1.0)
        input_embeds.add_(user_stream)

        loss_mask = torch.ones_like(
            torch.cat([text_labels.unsqueeze(-1), audio_labels], dim=-1),
            device=self.device,
            dtype=torch.bool,
        )


        return {
            "input_embeds": input_embeds,
            "input_lens": source_encoded_lens - 1,
            "output_lens": source_encoded_lens - 1,
            "text_labels": text_labels,
            "input_audio_tokens": audio_inputs,
            "audio_labels": audio_labels,
            "loss_mask": loss_mask,
            "speech_feat": speech_feat,
            # for flow matching training
            "speech_feat_len": speech_feat_lens,
            "speech_token": speech_tokens,
            "speech_token_len": speech_tokens_lens,
            "embedding": torch.zeros(speech_tokens.size(0), 192).to(self.device)
        }


    def cal_acc(self, pad_outputs, pad_targets, ignore_label):
        pad_pred = pad_outputs.argmax(-1)
        mask = pad_targets != ignore_label
        numerator = torch.sum(
            pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
        denominator = torch.sum(mask)
        return (numerator / denominator).detach().item()

    def training_step(self, batch: dict, batch_idx: int):
        # for m in (self.perception.preprocessor, self.perception.encoder, self.llm, self.speech_generation):
        #     if is_frozen(m):
        #         m.eval()
        inputs = self.prepare_inputs(batch)



        loss = self.audio_decoder.flow(inputs, self.device)

        ans = {
            "loss": loss['loss'],
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
        }
        self.log_dict(ans, on_step=True)
        return ans
        # import pdb; pdb.set_trace()


        forward_outputs = self(
            inputs["input_embeds"],
            input_audio_tokens=inputs["input_audio_tokens"],
            text_label=inputs['text_labels'],
            loss_mask=inputs["loss_mask"],
        )
        num_frames = inputs["input_lens"].sum()
        with loss_parallel():
            text_loss = (
                torch.nn.functional.cross_entropy(
                    forward_outputs["text_logits"].flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                    inputs["text_labels"].flatten(0, 1),
                    reduction="sum",
                )
                / num_frames
            )
            audio_loss = torch.nn.functional.cross_entropy(
                forward_outputs["audio_logits"].flatten(0, 2),  # (B, T, K, Vs) -> (*, Vs)
                inputs["audio_labels"].flatten(0, 2),
                reduction="sum",
            ) / (num_frames * self._num_codebooks)
        loss = self.cfg.text_loss_weight * text_loss + self.cfg.audio_loss_weight * audio_loss

        text_acc = self.cal_acc(forward_outputs["text_logits"].flatten(0, 1), inputs["text_labels"].flatten(0, 1), 0 )
        audio_acc = self.cal_acc(forward_outputs["audio_logits"].flatten(0, 1), inputs["audio_labels"].flatten(0, 1), -1)

        B, T = inputs["input_embeds"].shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "text_loss": text_loss,
            "audio_loss": audio_loss,
            "text_acc": text_acc,
            "audio_acc": audio_acc,
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "padding_ratio": num_frames / (B * T),
        }
        self.log_dict(ans, on_step=True)
        return ans

    def on_validation_epoch_start(self) -> None:
        self.on_train_epoch_start()
        self.asr_bleu = ASRBLEU(self.cfg.scoring_asr).reset()
        # self.bleu = BLEU().reset()
        self.mos = MOS().reset()

    def on_validation_epoch_end(self, prefix="val") -> None:
        asr_bleu = self.asr_bleu.compute()
        for k, m in asr_bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        # bleu = self.bleu.compute()
        # for k, m in bleu.items():
        #     self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

        mos = self.mos.compute()
        for k, m in mos.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)


    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            inputs = self.prepare_inputs(dataset_batch)

            batch = inputs['speech_token'].shape[0]


            this_uuid = str(uuid.uuid4())

            prompt_speech_feat = torch.zeros(batch, 0, 80).to(self.device)
            flow_prompt_speech_token = torch.zeros(batch, 0, dtype=torch.int64).to(self.device)
            spk_emb = torch.zeros(batch, 192).to(self.device)


            with fp32_precision(), torch.no_grad():

                response_speech, _ = self.audio_decoder.token2wav(inputs['speech_token'],
                                                                  uuid=this_uuid,
                                                                  prompt_token=flow_prompt_speech_token.to(self.device),
                                                                  prompt_feat=prompt_speech_feat.to(self.device),
                                                                  embedding=spk_emb,
                                                                  finalize=True)



                os.makedirs(self.cfg.get('audio_save_path'), exist_ok=True)
                for i in range(batch):
                    torchaudio.save(f"{self.cfg.audio_save_path}/{name}_{i}_{dataset_batch['sample_id'][i]}.wav",
                                    response_speech[i].unsqueeze(0).cpu(), 22050)

                pred_audios = resample(response_speech, 22050, 16000)

                self.asr_bleu.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    pred_audio=pred_audios,
                    pred_audio_lens=torch.tensor(pred_audios.shape[1] / 22050 * 16000).repeat(batch).to(torch.long),
                )

                self.mos.update(
                    name=name,
                    pred_audios=pred_audios,
                    tmp_dir=os.path.join(self.cfg.get('audio_save_path'), "tmp"),
                )


            # results = self.offline_inference(
            #     dataset_batch["source_audio"],
            #     dataset_batch["source_audio_lens"],
            #     decode_audio=True,
            # )
            #
            # with fp32_precision():  # resample is fragile to bfloat16 default dtype
            #     predicted_audio = resample(results['audio'], 22050, 16000)
            #     self.asr_bleu.update(
            #         name=name,
            #         refs=dataset_batch["target_texts"],
            #         pred_audio=predicted_audio,
            #         pred_audio_lens=(results["audio_len"] / 22050 * 16000).to(torch.long),
            #     )
            # self.bleu.update(name=name, refs=dataset_batch["target_texts"], hyps=results["text"])
            #
            # if self.cfg.get('audio_save_path') is not None and dist.get_rank() == 0:
            #
            #
            #     os.makedirs(self.cfg.get('audio_save_path'), exist_ok=True)
            #     logging.info(f"The shape of generated speech: {predicted_audio.shape}")
            #     for i in range(len(predicted_audio)):
            #         pred_audio = predicted_audio[i]
            #         user_audio = dataset_batch["source_audio"][i]
            #
            #         T1, T2 = pred_audio.shape[0], user_audio.shape[0]
            #         max_len = max(T1, T2)
            #         pred_audio_padded = torch.nn.functional.pad(pred_audio, (0, max_len - T1), mode='constant', value=0)
            #         user_audio_padded = torch.nn.functional.pad(user_audio, (0, max_len - T2), mode='constant', value=0)
            #
            #         result_audio = pred_audio_padded + user_audio_padded
            #
            #         torchaudio.save(f"{self.cfg.audio_save_path}/{name}_{i}_{dataset_batch['sample_id'][i]}.wav",
            #                         result_audio.unsqueeze(0).float().cpu(),
            #                         16000)
            #
            # dist.barrier()

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def _get_bos_embedding(self) -> torch.Tensor:
        """
        Remove the audio codec embedding for the beginning of AR decoding.
        """
        text_bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        input_embeds = self.embed_tokens(text_bos)
        return input_embeds

    @torch.no_grad()
    def offline_inference(
        self,
        input_signal: torch.Tensor,
        input_signal_lens: torch.Tensor,
        decode_audio: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive prediction.

        Args:
            input_signal: a batch of waveforms with shape (B, T) with source sampling rate.
            input_signal_lens: example lengths as number of samples of shape (B,).
            decode_audio: bool, whether to decode audio codes to waveform.

        Returns:
            A dict with keys:
                * "text": generated text, de-tokenized to strings, properly skipping text_pad_id; list of length B.
                * "tokens_text": generated text tokens of shape (B, T2).
                * "tokens_audio": generated audio codes of shape (B, T2, K) where `K=num_codebooks`.
                * "tokens_len" output lengths as number of tokens of shape (B,).
                * "audio": generated waveform of shape (B, T3) (`decode_audio=True`).
                * "audio_len" output lengths as number of waveform samples of shape (B,) (when `decode_audio=True`).
        """
        input_embeds, lengths = self.perception(
            input_signal=input_signal,
            input_signal_length=input_signal_lens,
        )

        # source_audio_tokens = extract_speech_token(
        #     self.whispervq,
        #     self.feature_extractor,
        #     [(input_signal[i].unsqueeze(0), 16000) for i in range(input_signal.shape[0])],
        # )

        # source_audio_tokens = torch.tensor(source_audio_tokens, dtype=torch.long, device=self.device)
        # source_audio_embeds = self.embed_audio_tokens(source_audio_tokens)
        # T_min = min(input_embeds.shape[1], source_audio_embeds.shape[1])

        # input_embeds = input_embeds[:, :T_min]
        # source_audio_embeds = source_audio_embeds[:, :T_min]


        B, T_local, H = input_embeds.shape

        # Determine decoding length and pad if FSDP
        if self._use_fsdp:
            T_tensor = torch.tensor([T_local], device=input_embeds.device)
            dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
            T = int(T_tensor.item())
            if T > T_local:
                last_frame = input_embeds[:, T_local - 1 : T_local, :]  # (B,1,H)
                pad = last_frame.repeat(1, T - T_local, 1)  # (B, T-T_local, H)
                input_embeds = torch.cat([input_embeds, pad], dim=1)
        else:
            T = T_local

        # Apply channel weight

        input_embeds = input_embeds * self.cfg.get("duplex_user_emb_weight", 1.0)

        # This cache is for self.llm
        cache = DynamicCache()
        # Call reset_input_and_kv_cache to enable cache for TransformerARSpeechDecoder
        self.speech_generation.reset_input_and_kv_cache(use_cache=True)
        gen_text = torch.empty(B, T, device=self.device, dtype=torch.long)
        gen_audio = torch.empty(B, T, self._num_codebooks, device=self.device, dtype=torch.long)

        # First step, use speech_delay token
        input_embeds[:, 0] += self._get_bos_embedding()



        first_audio = torch.full(
            [B, 1, self._num_codebooks],
            fill_value=self.speech_delay_id,
            device=self.device,
            dtype=torch.long,
        )
        ans = self(input_embeds[:, :1], cache=cache, input_audio_tokens=first_audio, loss_mask=None)
        gen_text[:, 0] = ans["text_logits"][:, -1].argmax(dim=-1)
        gen_audio[:, 0] = ans["audio_logits"][:, -1].argmax(dim=-1)

        # Autoregressive loop
        for t in range(1, T):
            last_emb = self.embed_tokens(gen_text[:, t - 1])
            input_embeds[:, t] += last_emb
            current_audio = gen_audio[:, t - 1 : t, :]
            ans = self(input_embeds[:, t : t + 1], cache=ans["cache"], input_audio_tokens=current_audio)
            gen_text[:, t] = ans["text_logits"][:, -1].argmax(dim=-1)
            gen_audio[:, t] = ans["audio_logits"][:, -1].argmax(dim=-1)

        # Trim back to local length if padded
        if self._use_fsdp and T > T_local:
            gen_text = gen_text[:, :T_local]
            gen_audio = gen_audio[:, :T_local]

        ans = {
            "text": tokens_to_str(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.text_pad_id),
            "tokens_text": gen_text,
            "tokens_audio": gen_audio,
            "tokens_len": lengths,
        }


        if decode_audio:
            # self.load_flow_decoder()
            with fp32_precision(), torch.no_grad():
                this_uuid = str(uuid.uuid4())

          
                prompt_speech_feat = torch.zeros(input_embeds.shape[0], 0, 80).to(self.device)
                flow_prompt_speech_token = torch.zeros(input_embeds.shape[0], 0, dtype=torch.int64).to(self.device)
                spk_emb = torch.zeros(input_embeds.shape[0], 192).to(self.device)

                flow_input_token = gen_audio[:,:,0]
                flow_input_token[flow_input_token >16383] = 0
                #
                response_speech, _ = self.audio_decoder.token2wav(flow_input_token,
                                                                  uuid=this_uuid,
                                                                  prompt_token=flow_prompt_speech_token.to(self.device),
                                                                  prompt_feat=prompt_speech_feat.to(self.device),
                                                                  embedding=spk_emb,
                                                                  finalize=True)

                # response_speech = self.audio_decoder.stream_inference(flow_input_token,
                #                                                   this_uuid,
                #                                                   flow_prompt_speech_token.to(self.device),
                #                                                   prompt_speech_feat.to(self.device),
                #                                                   spk_emb,
                #                                                   )
                ans["audio"] = response_speech
                ans["audio_len"] = torch.tensor(response_speech.shape[1]).unsqueeze(0).repeat(input_embeds.shape[0]).to(self.device)

        
        return ans

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        return configure_optimizers(self)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        return {
            "cls": dict,
            "inputs": [
                {"name": "source_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "source_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {"name": "target_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "target_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "target_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.tokenizer.vocab_size,
                },
            ],
        }

    def configure_model(self) -> None:
        # TODO(pzelasko): refactor into separate module re-usable across models
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.llm
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            for transformer_block in llm.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: "
                            f"set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                parallelize_module(transformer_block, tp_mesh, plan)

            for m in (self.lm_head, self.audio_head):
                parallelize_module(
                    m,
                    tp_mesh,
                    ColwiseParallel(
                        input_layouts=Shard(1),
                        output_layouts=Shard(-1),
                        use_local_output=False,
                    ),
                )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1
            self._use_fsdp = True

            fsdp_config = {"mesh": dp_mesh}

            for idx, layer in enumerate(llm.layers):
                llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.lm_head = fully_shard(self.lm_head, **fsdp_config)
            self.perception = fully_shard(self.perception, **fsdp_config)
            self.speech_generation = fully_shard(self.speech_generation, **fsdp_config)
