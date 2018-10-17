import torch
import argparse
#from model.1dcnn import DCNN
try:
    from .utils import data_load
    from .model.kernel import CNNGaussianKernel
    from .model.density_ratio import KDR
    from .metrics import f1_score
except:
    from utils import data_load
    from model.kernel import CNNGaussianKernel
    from model.density_ratio import KDR
    from metrics import f1_score
import torch.optim as optim
import numpy as np
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
except:
    plt = None

def train(n_epochs, data_loader, kdr, device, val_loader=None, save_path=None):
    # n : how many windows to be grouped? batch_size should be multiple of n
    batch_size  = data_loader.batch_size
    optimizer = optim.Adam(kdr.parameters(), lr=0.0005)
    step_size = 10
    decay_factor = 0.9
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma=decay_factor)
    best_val_score = -np.inf
    for i_epoch in range(n_epochs):
        scheduler.step()
        curr_lr = scheduler.get_lr()
        print(i_epoch, "-th epoch (learning_rate : {})".format(curr_lr))
        all_score = []
        all_loss = []
        all_true_y = []
        for X_ref, X_test, y in tqdm(data_loader):
            X_ref = X_ref.to(device)
            X_test = X_test.to(device)
            # n = X_test.shape[1]
            # torch.randperm(n)
            # TODO randomly sample X_center for large "n"
            J, loss = kdr(X_ref, X_test, None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_score.append(J.detach().cpu())
            all_loss.append(loss.detach().cpu())
        all_score = torch.cat(all_score, dim=0).numpy()
        all_loss = torch.stack(all_loss).mean().cpu().numpy()
        print("Training loss", all_loss)

        # validation
        if val_loader is not None:
            print("============================== start validation")
            all_score = []
            all_loss = []
            all_true_y = []
            for val_X_ref, val_X_test, val_y in val_loader:
                val_X_ref = val_X_ref.to(device)
                val_X_test = val_X_test.to(device)
                val_J, val_loss = kdr(val_X_ref, val_X_test, None)
                all_score.append(val_J.detach())
                all_loss.append(val_loss.detach())
                all_true_y.append(val_y.detach())
            all_score = torch.cat(all_score, dim=0).cpu().numpy()
            all_true_y = torch.cat(all_true_y, dim=0).cpu().numpy()
            all_loss = torch.stack(all_loss).mean().cpu().numpy()
            print("Validation loss", all_loss)

            thrs_l = np.linspace(0, 6, num=50)
            window = 6
            f1 = np.amax([f1_score(all_score, all_true_y, thrs, window)[2] for thrs
                    in thrs_l])
            print("F1 score", f1)
            if f1 > best_val_score and save_path is not None:
                torch.save(kdr.state_dict(), save_path)
                best_model_loss = all_loss
                best_model_score = all_score
                best_model_f1 = f1
            # matplotlib 으로 그리기
            try:
                plt.clf()
                plt.cla()
                plt.close()
                fig = plt.figure(figsize=(30, 4.8))
                plt.plot(np.arange(len(all_score)), all_score, zorder=1)
                plt.scatter(np.where(all_true_y == 1)[0], all_true_y[all_true_y== 1],
                    marker='x', c='r', zorder=2)
                plt.pause(0.001)
                plt.ion()
                plt.show()
                #plt.gcf().clear()
            except:
                pass

def main(conf):
    # Data Load
    train_loader, val_loader, window, input_dim = \
        data_load(filename='../N1Lounge8F_06/n1lounge8f_06_10sec.csv',
            validation_split=conf['validation_split'],
            window=conf['window'], jump=conf['jump'], n=conf['group_size'],
            batch_size=conf['batch_size']) # jump * n / 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Model Construct
    embedding_dim = conf['embedding_dim']
    conv_kernel = conf['conv_kernel']
    n_channels = conf['n_channels']#[50, 30, 10]
    conv_stride = conf['conv_stride']
    pool_kernel = conf['pool_kernel']
    length_scale = conf['length_scale']
    kernel = CNNGaussianKernel(input_dim, window, embedding_dim, conv_kernel,
        n_channels, conv_stride, pool_kernel, length_scale)
    alpha = conf['alpha']
    #lambda_list = [10**i for i in [-3, -2, -1, 0, 1]]
    reg_lambda = conf['reg_lambda']
    kdr = KDR(alpha, kernel, reg_lambda=reg_lambda).to(device)
    #data = pandas.read_csv("../../N1Lounge8F/n1lounge8f_06_1.csv")
    #cols = data.columns
    n_epochs = conf['n_epochs']
    train(n_epochs, train_loader, kdr, device, val_loader,
    save_path=conf['save_path'])
    #x_datetime = data['timestamp'].values[30:10037]
    #x_datetime = [np.datetime64(datetime.fromtimestamp(x), 's') for x in x_datetime]

    #y = np.load("score.npy")
    #print (x_datetime[2332])
    #task_label = pandas.read_csv("../../N1Lounge8F/Task_201806.csv", header=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--validation_split', type=float, default=0.5)
    parser.add_argument('--window', type=int, default=60,
            help='data size')
    parser.add_argument('--jump', type=int, default=3,
            help='step size to jump')
    parser.add_argument('--group_size', type=int, default=50,
            help='n')
    parser.add_argument('--batch_size', type=int, default=100,
            help='batch_size')
    # CNN
    parser.add_argument('--embedding_dim', type=int, default=30,
            help='CNN embedding dimension')
    parser.add_argument('--conv_kernel', type=int, default=5,
            help='1d CNN layer kernel size')
    parser.add_argument('--n_channels', type=int, default=[50, 30, 10],
            nargs='+')
    parser.add_argument('--conv_stride', type=int, default=2,
            help='1D CNN stride')
    parser.add_argument('--pool_kernel', type=int, default=1,
            help='pooling layer kernel size (== stride)')
    # Kernel
    parser.add_argument('--length_scale', type=int, default=1,
            help='kernel length scale')
    parser.add_argument('--alpha', type=float, default=0.01,
            help='to control the weight of reference distribution for RuLSIF')
    parser.add_argument('--reg_lambda', type=float, default=0.1,
            help='regularization parameter to prevent overfitting for kernel \
            model parameter theta')
    # Training
    parser.add_argument('--n_epochs', type=int, default=1000,
            help='epochs')

    conf = vars(parser.parse_args())
    conf['save_path'] = 'best_param.model'
    main(conf)
