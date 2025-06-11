import os
import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf
import torch.distributed.checkpoint as dist_cp
from nemo.collections.speechlm2 import DataModule, DuplexS2SDataset, DuplexS2SSpeechDecoderModel
from nemo.core.config import hydra_runner


def load_with_manual_handling(cfg, checkpoint_path):
    # 检查meta.pt文件获取模型配置
    meta_path = os.path.join(checkpoint_path, "meta.pt")
    if os.path.exists(meta_path):
        meta_info = torch.load(meta_path, map_location="cpu")
        if isinstance(meta_info, dict) and "hyper_parameters" in meta_info:
            model_config = meta_info["hyper_parameters"]
            model = DuplexS2SSpeechDecoderModel(**model_config)
        else:
            model = DuplexS2SSpeechDecoderModel(OmegaConf.to_container(cfg.model, resolve=True))
    else:
        model = DuplexS2SSpeechDecoderModel(OmegaConf.to_container(cfg.model, resolve=True))

    # 读取checkpoint元数据
    reader = dist_cp.FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()

    # 获取所有checkpoint键
    checkpoint_keys = set(metadata.state_dict_metadata.keys())

    # 构造符合dist_cp要求的state_dict
    target_state_dict = {}
    for key in checkpoint_keys:
        # 去掉state_dict前缀，匹配模型键
        model_key = key.replace("state_dict.", "")

        if model_key in model.state_dict():
            # 创建与模型参数相同形状的空tensor（必须在CPU上）
            model_param = model.state_dict()[model_key]
            target_state_dict[key] = torch.empty_like(model_param, device="cpu")

    if not target_state_dict:
        return None

    # 加载分片checkpoint
    dist_cp.load_state_dict(
        state_dict=target_state_dict,
        storage_reader=reader,
        no_dist=True,
    )

    # 提取并转换到模型键
    model_state_dict = {}
    for key, value in target_state_dict.items():
        model_key = key.replace("state_dict.", "")
        model_state_dict[model_key] = value.to(model.device)

    # 加载到模型
    result = model.load_state_dict(model_state_dict, strict=False)

    return model

@hydra_runner(config_path="conf", config_name="s2s_duplex_speech_decoder")
def inference(cfg):
    OmegaConf.resolve(cfg)

    # 设置单GPU trainer
    trainer = Trainer(
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="auto"  # 不使用FSDP
    )

    # 直接使用Lightning的load_from_checkpoint
    checkpoint_path = "/lustre/fsw/portfolios/llmservice/users/cchen1/code/duplex_s2s/exp/train_llama_7b_4node_decoder_all_3-4/checkpoints/step=16005.ckpt"
    model = load_with_manual_handling(cfg, checkpoint_path)
    model.eval()


    dataset = DuplexS2SDataset(
        tokenizer=model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        target_sample_rate=cfg.data.target_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
    )
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)
    datamodule.setup("validate")

    trainer.validate(model, datamodule)
    import pdb; pdb.set_trace()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    inference()
