import os
import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2.data import DuplexS2SDataset
from nemo.collections.speechlm2.data.datamodule import DataModule
from nemo.collections.speechlm2.models import DuplexS2SSpeechDecoderModel
from nemo.core.config import hydra_runner
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
torch.set_float32_matmul_precision("high")

@hydra_runner(config_path="conf", config_name="s2s_duplex_speech_decoder")
def infer(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))

    # 只初始化模型，不加载 train data
    model = DuplexS2SSpeechDecoderModel(OmegaConf.to_container(cfg.model, resolve=True))
    dataset = DuplexS2SDataset(
        tokenizer=model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        target_sample_rate=cfg.data.target_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
    )
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    # 指定 checkpoint 路径（如 "best" 或具体路径）

    ckpt_path = cfg.exp_manager.resume_from_checkpoint

    # 执行 validation，只调用 validation dataloader
    trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == "__main__":
    infer()
