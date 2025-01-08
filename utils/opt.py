import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

def get_opts():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset_dir', type=str, default='../data/251212_G2_Rev_Accumulate_v3/')
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--meta_path', type=str, default='./data')

    # dataloader
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--resolution', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_list', type=str, default="train")
    parser.add_argument('--val_list', type=str, default="val")
    parser.add_argument('--test_list', type=str, default="test")

    # model
    parser.add_argument('--model_name', type=str, default="imagenet")
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--depth', type=int, default=50)
    parser.add_argument('--bottleneck', type=bool, default=True)
    parser.add_argument('--resume_weights', type=str, help='resume_weights')
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--patch_num', type=int, default=16)
    parser.add_argument('--overlap', type=int, default=100)

    # train
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=40)
    parser.add_argument('--gamma', type=float, default=0.1)

    # Balanced Contrastive Learning
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for BCL')
    parser.add_argument('--beta', type=float, default=0.35, help='beta for BCL')
    
    # image preprocessing
    parser.add_argument('--image_cut', type=str, default='crop')
    parser.add_argument("--h_flip", action='store_true', default=True)
    parser.add_argument("--v_flip", action='store_true', default=True)
    
    # loss
    parser.add_argument('--loss_function', type=str)

    # pretext task (lorot)
    parser.add_argument('--pretext', type=str, default='None')
    parser.add_argument('--pretext_ratio', type=float, default=0.1)

    # fg-ssl task
    parser.add_argument('--fgssl', type=bool, default=False)
    parser.add_argument("--patches", type = int, nargs = "*")
    parser.add_argument("--featdim", type = int, default = 512, required = False)
    parser.add_argument("--momentum", type = float, default = 0.9, required = False)
    parser.add_argument("--cut_prob", type = float, default = -1.1, required = False)
    parser.add_argument("--beta2", type = int, default = 1, required = False)

    # dino
    parser.add_argument('--concat', action='store_true', default=False)
    parser.add_argument('--lr_list', nargs="+", type=float, default=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--last_blocks_list", nargs="+", type=int, default=[1, 4])
    parser.add_argument('--avg_pool_list', nargs="+", type=str2bool, default=[False, True])

    # etc
    parser.add_argument('--gpus', nargs="+", type=int, default=[0])
    parser.add_argument('--exp_name', type=str, default='exp')

    # evaluation
    parser.add_argument('--eval', action='store_true', default=False)
    return parser.parse_args()
