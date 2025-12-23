import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from networks.cascade.cnn_vit_backbone import CONFIGS as CONFIGS_ViT_seg
from networks.cascade.networks import TransCASCADE
from networks.ts2_kv_rope_fft import TranXnet
from dataset_synapse import Synapse_dataset
from datasets.dataset_acdc import BaseDataSets

from utils_volin import test_single_volume
# from utils_nonorm import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument(
    "--volume_path",
    type=str,
    default="/home/user/my_code/Synapse_npy/test/images/",
    help="root dir for validation volume data",
)  # for acdc volume_path=root_dir
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument("--checkpoint_path", type=str, default="/data/xb_model/new_frame_model/transEncoder_kv_ffn_rope_epoch_199.pth", help="checpoint to load")
parser.add_argument("--num_classes", type=int, default=9, help="output channel of network")
parser.add_argument("--list_dir", type=str, default="/home/user/my_code/new_flame/deformableLKA-main/2D/lists/lists_Synapse", help="list dir")
parser.add_argument("--output_dir", type=str, default="/home/user/my_code/new_flame/deformableLKA-main/2D/model_out_ab_sy", help="output dir")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
parser.add_argument("--max_epochs", type=int, default=400, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
parser.add_argument("--model_name", type=str, default="Cascade", help="model_name")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
parser.add_argument("--is_savenii", action="store_true", help="whether to save results during inference")
parser.add_argument("--test_save_dir", type=str, default="../predictions", help="saving prediction as nii!")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation network learning rate")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs="+",
)
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
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')  

args = parser.parse_args()
# if args.dataset == "Synapse":
#     args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
# config = get_config(args)


def inference(model, test_save_path=None,device=None,root_path=None,list_dir=None,model_name=None):
    logging.basicConfig(
        filename=test_save_path+ '/log_' + model_name  + ".txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", img_size=args.img_size, list_dir=args.list_dir)
    db_test = Synapse_dataset(base_dir=root_path, split="test_vol", list_dir=list_dir, img_size=224)
    # db_test = args.Dataset(base_dir=root_path, split="test_vol", img_size=args.img_size, list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    all_slice_mean_dices = []
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch["case_name"][0]
        metric_i,slice_dices = test_single_volume(
            image,
            label,
            model,
            classes=9,
            patch_size=[224, 224],
            test_save_path=None,
            case=case_name,
            z_spacing=1,
            device=device,
            return_slice_metrics=True
        )
        metric_list += np.array(metric_i)
            
        all_slice_mean_dices.extend(list(slice_dices))

        logging.info(
            "idx %d case %s mean_dice %f mean_hd95 %f"
            % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1])
        )
    metric_list = metric_list / len(db_test)
    for i in range(1, 9):
        logging.info("Mean class %d mean_dice %f mean_hd95 %f" % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info("Testing performance in best val model: mean_dice : %f mean_hd95 : %f" % (performance, mean_hd95))

    out_dir = test_save_path if test_save_path is not None else './'
    os.makedirs(out_dir, exist_ok=True)
    slice_npy_path = os.path.join(out_dir, f"{model_name}_mean_dice.npy")
    np.save(slice_npy_path, np.array(all_slice_mean_dices, dtype=np.float32))
    logging.info(f"Saved per-slice mean-dice array to {slice_npy_path}")
    return "Testing Finished!"


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    dataset_config = {
        "Synapse": {
            "Dataset": Synapse_dataset,
            'volume_path': args.volume_path,
            "z_spacing": 1,
            "num_classes": 9,
            "pth_path": '/data/xb_model/compare_model/ACDC/EMCAD/CASCADE_sy/best.pth',
            'output_dir': '/home/user/my_code/new_flame/deformableLKA-main/2D/model_out',
            'list_dir': '/home/user/my_code/new_flame/deformableLKA-main/2D/lists/lists_Synapse'
        },
        "ACDC":
        {
            "Dataset": BaseDataSets,
            'volume_path': "/data/xb_dataset/ACDC",
            "z_spacing": 1,
            "num_classes": 4,
            "pth_path":'/data/xb_model/compare_model/ACDC/EMCAD/PGRUnet/best.pth',
            'output_dir': '/home/user/my_code/new_flame/deformableLKA-main/2D/model_out_ACDC'
            ,'list_dir': '/home/user/my_code/deformableLKA-main/2D/lists/lists_ACDC/'
        },
    }
    dataset_name = args.dataset
    args.Dataset = dataset_config[dataset_name]["Dataset"]
    args.volume_path = dataset_config[dataset_name]["volume_path"]
    args.z_spacing = dataset_config[dataset_name]["z_spacing"]
    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.checkpoint_path = dataset_config[dataset_name]["pth_path"]
    args.output_dir = dataset_config[dataset_name]["output_dir"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]
    args.is_pretrain = True

    # net = TranXnet(in_chan=3,base_chan=32,num_classes=args.num_classes).to(device)
    # net.load_state_dict(torch.load(args.checkpoint_path,map_location=device))
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    net = TransCASCADE(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)
    net.load_state_dict(torch.load(args.checkpoint_path, map_location=device))

    
    log_folder = args.output_dir
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=log_folder+ '/log_' +args.model_name + args.dataset + ".txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    test_save_path = None
    inference(args, net, test_save_path,device=device)
