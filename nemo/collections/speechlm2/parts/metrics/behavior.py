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
from collections import defaultdict

import sacrebleu
import torch
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_nemo
from nemo.utils import logging


class BEHAVIOR:
    """
    Computes BLEU scores on
    ASR predictions on generated audio with pretrained NeMo ASR.
    By default, uses Whisper's EnglishTextNormalizer on hypotheses and references.
    """

    def __init__(self) -> None:


        self.turn_taking_latency = defaultdict(list)
        self.turn_taking_acc = defaultdict(list)
        self.barge_in_latency = defaultdict(list)
        self.barge_in_acc = defaultdict(list)

    def reset(self) -> None:

        torch.cuda.memory.empty_cache()
        with fp32_precision():  # Some NeMo ASR models weren't trained with bfloat16.
            self.vad, utils = torch.hub.load('snakers4/silero-vad', model='silero_vad',  trust_repo=True, force_reload=False)
        self.get_speech_timestamps, _, _, _, _ = utils

        return self

    def find_all_overlaps(self, user_turns, agent_turns):

        i, j = 0, 0
        overlaps = []

        while i < len(user_turns) and j < len(agent_turns):
            u_start, u_end = user_turns[i]['start'], user_turns[i]['end']
            a_start, a_end = agent_turns[j]['start'], agent_turns[j]['end']

            overlap_start = max(u_start, a_start)
            overlap_end = min(u_end, a_end)

            if overlap_start < overlap_end:
                overlaps.append(overlap_end - overlap_start)

            if u_end < a_end:
                i += 1
            else:
                j += 1
        return overlaps

    def update(
        self, name: str, pred_audio: torch.Tensor, user_audio: torch.Tensor = None,
    ) -> list[str]:
        self.vad.to(pred_audio.device)
        batch = pred_audio.size(0)

        tt_latency, tt_result, bi_seg  = [], [], []
        for i in range(batch):
            user_timestamps = self.get_speech_timestamps(user_audio[i], self.vad, sampling_rate=16000, min_silence_duration_ms=1500)
            user_timestamps = [{'start': s['start'] / 16000, 'end': s['end'] / 16000} for s in user_timestamps]
            agent_timestamps = self.get_speech_timestamps(pred_audio[i], self.vad, sampling_rate=16000, min_silence_duration_ms=1500)
            agent_timestamps = [{'start': s['start'] / 16000, 'end': s['end'] / 16000} for s in agent_timestamps]

            if len(agent_timestamps) > 0:

                tt_latency_i = agent_timestamps[0]['start'] - user_timestamps[0]['end']

                if 0 < tt_latency_i < 1.6:
                    tt_result.append(1)
                    tt_latency.append(tt_latency_i)
                else:
                    tt_result.append(0)


            bi_seg.extend(self.find_all_overlaps(user_timestamps, agent_timestamps))


        if len(bi_seg) > 0:
            self.barge_in_latency[name].append(sum(bi_seg) / len(bi_seg))
            self.barge_in_acc[name].append(sum(i < 1.5 for i in bi_seg)  / len(bi_seg))

        if len(tt_latency) > 0:
            self.turn_taking_latency[name].append(sum(tt_latency) / len(tt_latency))

        if len(tt_result) > 0:
            self.turn_taking_acc[name].append(sum(tt_result) / len(tt_result))


    def compute(self) -> dict[str, torch.Tensor]:

        corpus_metric = {}
        for name in self.turn_taking_latency.keys():
            tt_acc = torch.tensor(self.turn_taking_acc[name])
            corpus_metric[f"turn_taking_acc_{name}"] = tt_acc.mean()

            tt_latency = torch.tensor(self.turn_taking_latency[name])
            corpus_metric[f"turn_taking_latency_{name}"] = tt_latency.mean()


            if len(self.barge_in_acc[name]) > 0:
                bi_acc = torch.tensor(self.barge_in_acc[name])
                corpus_metric[f"barge_in_acc_{name}"] = bi_acc.mean()

                bi_latency = torch.tensor(self.barge_in_latency[name])
                corpus_metric[f"barge_in_latency_{name}"] = bi_latency.mean()


        # corpus_metric['turn_taking_acc'] = torch.stack(list(corpus_metric.values())).mean()

        self.turn_taking_latency.clear()
        self.turn_taking_acc.clear()
        self.barge_in_latency.clear()
        self.barge_in_acc.clear()

        self.vad = None  # free up GPU memory
        torch.cuda.memory.empty_cache()
        return corpus_metric


def _identity(x):
    return x
