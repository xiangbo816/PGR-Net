import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import logging
import random
import warnings
from pydoc import locate

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from trainer import trainer_synapse

from networks.PGR_Net import PGR_Net

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="/home/user/my_code/Synapse_npy/train/images/",
    help="root dir for train data",
)
parser.add_argument(
    "--test_path",
    type=str,
    default="/home/user/my_code/Synapse_npy/test/images/",
    help="root dir for test data",
)
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument("--list_dir", type=str, default="lists/lists_Synapse", help="list dir")
parser.add_argument("--num_classes", type=int, default=9, help="output channel of network")
parser.add_argument("--output_dir", type=str, default="model_out", help="output dir")
parser.add_argument("--model_dir", type=str, default="/data/xb_model/Synapse", help="model output dir")  
parser.add_argument("--log_dir", type=str, default="/data/xb_train_log", help="log output dir")
parser.add_argument("--max_iterations", type=int, default=50000, help="maximum epoch number to train")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
parser.add_argument("--model_name", type=str, default="PGR_Net", help="model_name")
parser.add_argument("--operation", type=str, default="学习率0.05,训练200轮,batch为24", help="operation")
parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
parser.add_argument("--num_gpu", type=int, default=0, help="use which gpu")
parser.add_argument("--max_epochs", type=int, default=200, help="maximum epoch number to train")
parser.add_argument("--eval_interval", type=int, default=15, help="eval_interval")
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation network base learning rate")
parser.add_argument("--deterministic", type=int, default=1, help="whether to use deterministic training")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
parser.add_argument("--z_spacing", type=int, default=1, help="z_spacing")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--zip", action="store_true", help="use zipped dataset instead of folder dataset")
parser.add_argument(
    "--cache-mode",
    type=str,
    default="part",
    choices=["no", "full", "part"],
    help="no: no cache, "
    "full: cache all data, "
    "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
)
parser.add_argument("--resume", help="resume from checkpoint")
parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
parser.add_argument(
    "--use-checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory"
)
parser.add_argument(
    "--amp-opt-level",
    type=str,
    default="O1",
    choices=["O0", "O1", "O2"],
    help="mixed precision opt level, if O0, no amp is used",
)
parser.add_argument("--tag", help="tag of experiment")
parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
parser.add_argument("--throughput", action="store_true", help="Test throughput only")
parser.add_argument(
    "--module", help="The module that you want to load as the network, e.g. networks.DAEFormer.DAEFormer"
)

args = parser.parse_args()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()
    
    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        "Synapse": {
            "root_path": args.root_path,
            "list_dir": args.list_dir,
            "num_classes": 9,
        },
    }

    if args.batch_size != 24 and args.batch_size % 5 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.root_path = dataset_config[dataset_name]["root_path"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    net = PGR_Net(in_chan=3,base_chan=32, num_classes=args.num_classes).to(device)


    # from torchinfo import summary
    # summary(
    #     net,
    #     input_size=(1,3, 224, 224),  # 输入维度 (C, H, W)
    #     col_names=["input_size", "output_size", "num_params", "kernel_size"],  # 显示更多信息
    #     verbose=1  # 完整输出模式
    # )

    trainer = {
        "Synapse": trainer_synapse,
    }
    trainer[dataset_name](args, net, args.output_dir,device=device)
