import os
import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf
import torch.distributed.checkpoint as dist_cp
from nemo.collections.speechlm2 import DataModule, DuplexS2SDataset, DuplexS2SSpeechDecoderModel
from nemo.core.config import hydra_runner


def load_with_key_mapping(cfg, checkpoint_path):
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


    reader = dist_cp.FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()

    checkpoint_keys = set(metadata.state_dict_metadata.keys())


    target_state_dict = {}
    for key in checkpoint_keys:
        model_key = key.replace("state_dict.", "")

        if model_key in model.state_dict():
            model_param = model.state_dict()[model_key]
            target_state_dict[key] = torch.empty_like(model_param, device="cpu")

    if not target_state_dict:
        return None

    dist_cp.load_state_dict(
        state_dict=target_state_dict,
        storage_reader=reader,
        no_dist=True,
    )


    model_state_dict = {}
    for key, value in target_state_dict.items():
        model_key = key.replace("state_dict.", "")
        model_state_dict[model_key] = value.to(model.device)


    model.load_state_dict(model_state_dict, strict=False)

    return model

@hydra_runner(config_path="conf", config_name="s2s_duplex_speech_decoder")
def inference(cfg):
    OmegaConf.resolve(cfg)


    trainer = Trainer(
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="auto"
    )

    checkpoint_path = cfg.exp_manager.resume_from_checkpoint
    model = load_with_key_mapping(cfg, checkpoint_path)
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
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    inference()
