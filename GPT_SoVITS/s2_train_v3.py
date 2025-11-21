import warnings

warnings.filterwarnings("ignore")
import os

import utils

hps = utils.get_hparams(stage=2)
os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace("-", ",")
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)
from random import randint

from module import commons
from module.data_utils import (
    DistributedBucketSampler,
)
from module.data_utils import (
    TextAudioSpeakerCollateV3 as TextAudioSpeakerCollate,
)
from module.data_utils import (
    TextAudioSpeakerLoaderV3 as TextAudioSpeakerLoader,
)
from module.models import (
    SynthesizerTrnV3 as SynthesizerTrn,
)
from process_ckpt import savee

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
###反正A100fp32更快，那试试tf32吧
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision(
    "medium"
)  # 最低精度但最快（也就快一丁点），对于结果造成不了影响
global_step = 0

device = "cpu"  # cuda以外的设备，等mps优化后加入


def main():
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))

    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.data.exp_dir)
        logger.info(hps)
        writer = SummaryWriter(log_dir=hps.s2_ckpt_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.s2_ckpt_dir, "eval"))

    dist.init_process_group(
        backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
        init_method="env://?use_libuv=False",
        world_size=n_gpus,
        rank=rank,
    )
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    train_dataset = TextAudioSpeakerLoader(hps.data)  ########
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [
            32,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
        ],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=6,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=4,
    )

    net_g = (
        SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).cuda(rank)
        if torch.cuda.is_available()
        else SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)
    )

    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),  ###默认所有层lr一致
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    if torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    else:
        net_g = net_g.to(device)

    try:  # 如果能加载自动resume
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(
                f"{hps.data.exp_dir}/logs_s2_{hps.model.version}", "G_*.pth"
            ),
            net_g,
            optim_g,
        )
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)
    except:  # 如果首次不能加载，加载pretrain
        epoch_str = 1
        global_step = 0
        if hps.train.pretrained_s2G not in {"", None} and os.path.exists(
            hps.train.pretrained_s2G
        ):
            if rank == 0:
                logger.info(f"loaded pretrained {hps.train.pretrained_s2G}")
            print(
                f"loaded pretrained {hps.train.pretrained_s2G}",
                net_g.module.load_state_dict(
                    torch.load(
                        hps.train.pretrained_s2G, map_location="cpu", weights_only=False
                    )["weight"],
                    strict=False,
                )
                if torch.cuda.is_available()
                else net_g.load_state_dict(
                    torch.load(
                        hps.train.pretrained_s2G, map_location="cpu", weights_only=False
                    )["weight"],
                    strict=False,
                ),
            )  ##测试不加载优化器

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=-1
    )
    for _ in range(epoch_str):
        scheduler_g.step()

    scaler = GradScaler(enabled=hps.train.fp16_run)

    net_d = optim_d = scheduler_d = None
    print(f"start training from epoch {epoch_str}")
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
            )
        scheduler_g.step()
    print("training done")


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    optims,
    schedulers,
    scaler,
    loaders,
    logger,
    writers,
):
    net_g, _net_d = nets
    optim_g, _optim_d = optims
    train_loader, _eval_loader = loaders
    if writers is not None:
        writer, _writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    for batch_idx, (
        ssl,
        spec,
        mel,
        ssl_lengths,
        spec_lengths,
        text,
        text_lengths,
        mel_lengths,
    ) in enumerate(tqdm(train_loader)):
        if torch.cuda.is_available():
            spec, spec_lengths = (
                spec.cuda(
                    rank,
                    non_blocking=True,
                ),
                spec_lengths.cuda(
                    rank,
                    non_blocking=True,
                ),
            )
            mel, mel_lengths = (
                mel.cuda(rank, non_blocking=True),
                mel_lengths.cuda(rank, non_blocking=True),
            )
            ssl = ssl.cuda(rank, non_blocking=True)
            ssl.requires_grad = False
            text, text_lengths = (
                text.cuda(
                    rank,
                    non_blocking=True,
                ),
                text_lengths.cuda(
                    rank,
                    non_blocking=True,
                ),
            )
        else:
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            mel, mel_lengths = mel.to(device), mel_lengths.to(device)
            ssl = ssl.to(device)
            ssl.requires_grad = False
            text, text_lengths = text.to(device), text_lengths.to(device)

        with autocast(enabled=hps.train.fp16_run):
            cfm_loss = net_g(
                ssl,
                spec,
                mel,
                ssl_lengths,
                spec_lengths,
                text,
                text_lengths,
                mel_lengths,
                use_grad_ckpt=hps.train.grad_ckpt,
            )
            loss_gen_all = cfm_loss
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [cfm_loss]
                logger.info(
                    f"Train Epoch: {epoch} [{100.0 * batch_idx / len(train_loader):.0f}%]"
                )
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "learning_rate": lr,
                    "grad_norm_g": grad_norm_g,
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict,
                )

        global_step += 1
    if epoch % hps.train.save_every_epoch == 0 and rank == 0:
        if hps.train.if_save_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(
                    f"{hps.data.exp_dir}/logs_s2_{hps.model.version}",
                    f"G_{global_step}.pth",
                ),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(
                    f"{hps.data.exp_dir}/logs_s2_{hps.model.version}",
                    f"G_{233333333333}.pth",
                ),
            )
        if rank == 0 and hps.train.if_save_every_weights:
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            logger.info(
                "saving ckpt {}_e{}:{}".format(
                    hps.name,
                    epoch,
                    savee(
                        ckpt,
                        hps.name + f"_e{epoch}_s{global_step}",
                        epoch,
                        global_step,
                        hps,
                    ),
                )
            )

    if rank == 0:
        logger.info(f"====> Epoch: {epoch}")


if __name__ == "__main__":
    main()
