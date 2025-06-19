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
import torchaudio
import torch
import utmosv2
import os
from nemo.utils import logging
from nemo.collections.speechlm2.parts.precision import fp32_precision

class MOS:
    """
    Computes MOS scores on audio predictions.
    By default, uses UTMOS2.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose


        self._score = defaultdict(list)

    def reset(self):
        torch.cuda.memory.empty_cache()
        with fp32_precision():  # Some NeMo ASR models weren't trained with bfloat16.
            self.model = utmosv2.create_model(pretrained=True)
        return self

    def update(self, name, pred_audios, tmp_dir) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        rank_tmp_dir = os.path.join(tmp_dir, f"rank_{local_rank}")
        os.makedirs(rank_tmp_dir, exist_ok=True)


        for i, waveform in enumerate(pred_audios):

            waveform = waveform.detach().cpu()
            torchaudio.save(f"{rank_tmp_dir}/sample_{i}.wav", waveform.unsqueeze(0), 16000)
            if i > 1:
                break

        torch.distributed.barrier()

        all_mos = self.model.predict(input_dir=rank_tmp_dir)

        mos_score = sum([item['predicted_mos'] for item in all_mos]) / len(all_mos)
        self._score[name].append(mos_score)

    def compute(self) -> dict[str, torch.Tensor]:
        corpus_metric = {}

        for name in self._score.keys():
            metric = torch.tensor(self._score[name])
            corpus_metric[f"mos_{name}"] = metric.mean()

        corpus_metric['mos'] = torch.stack(list(corpus_metric.values())).mean()
        self._score.clear()
        return corpus_metric


def _identity(x):
    return x
