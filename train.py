import argparse

from torch.utils.data import DataLoader
from datasets.ZSSR_dataset import ZSSRDataset
import torch
from visualizer import Visualizer   #not the major
from models.zero_shot_SR import ZeroShotSRModel

parser = argparse.ArgumentParser(description="Pytorch DAZSSR")
parser.add_argument("--dataroot", type=str, default='./datasets/ZSSR1', help="training dataset")
parser.add_argument("--batch-size", type=int, default=4, help="training batch size")
parser.add_argument('--crop_size', type=int, default=128, help='scale images to this size')
parser.add_argument('--sr_factor', type=int, default=2, help='super resolution scale')
parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', type=int, default=32, help='# of discrim filters in first conv layer')
parser.add_argument('--n_blocks_SR', type=int, default=5, help='only used if netD==n_layers')
parser.add_argument('--n_blocks_ST', type=int, default=2, help='only used if netD==n_layers')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument("--nEpochs", type=int, default=30, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=1e-4")
parser.add_argument('--name', type=str, default='ZSSR1',
                    help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--seed', type=int, default=456, help='random seed to use. Default=123')
parser.add_argument('--mean', type=list, default=(0.5, 0.5, 0.5), help='normalized mean')
parser.add_argument('--std', type=list, default=(0.5, 0.5, 0.5), help='normalized std')
parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
parser.add_argument('--print_freq', type=int, default=20, help='frequency of showing training results on console')
parser.add_argument('--save_images_freq', type=int, default=500, help='frequency of showing training results on console')
parser.add_argument('--save_epoch_freq', type=int, default=5,
                    help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--continue_train', type=bool, default=False, help='if True, continue train')
parser.add_argument("--epoch", type=int, default=0, help="continue train from epoch 50")
parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=500, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--niter', type=int, default=5, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=20, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--init_type', type=str, default='normal',
                    help='network initialization [normal|xavier|kaiming|orthogonal]')
parser.add_argument('--n_repeat', type=int, default=50, help='repeat n times training per epoch ')



def main():
    opt = parser.parse_args()

    torch.manual_seed(opt.seed)

    # set gpu_ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    opt.isTrain = True

    model = ZeroShotSRModel()
    dataset = ZSSRDataset(opt)
    model.initialize(opt)
    model.setup(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads)
    visualizer = Visualizer(opt)

    for epoch in range(opt.epoch, opt.nEpochs + 1):
        for k in range(opt.n_repeat):
            for i, data in enumerate(dataloader):
                model.set_input(data)
                model.optimize_parameters()

                if i % opt.display_freq == 0:
                    save_result = i % opt.save_images_freq == 0
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if i % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    visualizer.print_current_losses(epoch, i, losses)

                    visualizer.plot_current_losses(losses)

            if epoch % opt.save_epoch_freq == 0:
                model.save_networks(epoch)
            model.update_learning_rate()



if __name__ == '__main__':
    main()