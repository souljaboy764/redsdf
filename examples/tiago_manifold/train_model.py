import os
import time
from datetime import datetime
import argparse
import torch
import tqdm
import distutils.version
from torch.utils.tensorboard import SummaryWriter
from redsdf.models.redsdf import RegularizedDeepSignedDistanceFields
from redsdf.data_loader import construct_loader
from redsdf.early_stopping_tools import EarlyStopping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./data", help="path of save generated data")
    parser.add_argument('--log_dir', type=str, default="./logs", help="path to save logs")
    parser.add_argument('--use_cuda', action="store_true", default=False, help="whether to use cuda")
    parser.add_argument('--batch_size', type=int, default=4096, help="batch size to train")
    parser.add_argument('--center', type=str, default="0.0-0.0-0.5", help="center of object")
    parser.add_argument('--num_epochs', type=int, default=2000, help="number of epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate to train network")
    parser.add_argument('--patience', type=int, default=50, help="patience for early stopping")
    args = parser.parse_args()

    data_dir = args.data_dir
    log_dir = args.log_dir + "/exp-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    num_epochs = args.num_epochs
    device = 'gpu' if args.use_cuda else 'cpu'
    point_batch_size = args.batch_size
    num_workers = 4
    validate_per_epochs = 1
    gamma = 0.02

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    if device == "gpu" and torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = 'cpu'
    print('DEVICE = ', DEVICE)

    center = list(map(float, args.center.split('-')))

    train_loader, valid_loader, test_loader, poses = construct_loader(data_dir, batch_size=point_batch_size,
                                                                      pin_memory=True,
                                                                      num_workers=num_workers, shuffle=True)
    poses = poses.to(DEVICE)
    n_points_train = len(train_loader.dataset)
    n_points_validate = len(valid_loader.dataset)

    redsdf = RegularizedDeepSignedDistanceFields(input_dim=11,
                                                 hidden_sizes=[512, 512, 512, 512, 512],
                                                 output_dim=1,
                                                 center=center,
                                                 activation='relu',
                                                 device=DEVICE,
                                                 mode_switching_starting_layer=2,
                                                 mode_switching_hidden_sizes=32,
                                                 mode_switching_alpha_last_layer_nonlinear='softplus',
                                                 mode_switching_alpha_scale_center=[1, 1],
                                                 mode_switching_rho_last_layer_nonlinear='sigmoid',
                                                 mode_switching_rho_scale_bias=[1, 0.5],
                                                 mode_switching_sigma_activation='relu')
    opt = torch.optim.Adam(redsdf.nn_model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=args.patience, verbose=False, path=log_dir + "/checkpoint.pt")

    for i in tqdm.tqdm(range(num_epochs)):
        epoch = i + 1
        redsdf.train()
        train_loss = 0
        t_data_loader = 0.
        t_compute_loss = 0.
        t_compute_gradient = 0.
        t_start = time.time()
        for points in train_loader:
            points = points.to(redsdf.device)
            t_data_loader += time.time() - t_start
            opt.zero_grad(set_to_none=True)
            t_start = time.time()
            norm_level_loss, J_nspace_proj_loss, cov_nspace_proj_loss, deviation_regularization = \
                redsdf.get_loss_components(points, poses)
            loss = norm_level_loss + J_nspace_proj_loss.mean() + cov_nspace_proj_loss.mean() + \
                   gamma * deviation_regularization.mean()
            t_compute_loss += time.time() - t_start
            t_start = time.time()
            loss.backward()
            opt.step()
            t_compute_gradient += time.time() - t_start
            train_loss += loss.item() * points.shape[0]
            t_start = time.time()

        train_loss = train_loss / n_points_train
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Time/data_loader", t_data_loader, epoch)
        writer.add_scalar("Time/compute_loss", t_compute_loss, epoch)
        writer.add_scalar("Time/compute_gradient", t_compute_gradient, epoch)

        # Validation
        if epoch % validate_per_epochs == 0:
            redsdf.eval()
            validate_loss = 0
            t_start = time.time()

            for points in valid_loader:
                points = points.to(redsdf.device)
                norm_level_loss, J_nspace_proj_loss, cov_nspace_proj_loss, deviation_regularization = \
                    redsdf.get_loss_components(points, poses)
                loss = norm_level_loss + J_nspace_proj_loss.mean() + cov_nspace_proj_loss.mean() + \
                       gamma * deviation_regularization.mean()
                validate_loss += loss.item() * points.shape[0]

            validate_loss = validate_loss / n_points_validate
            writer.add_scalar("Loss/validate", validate_loss, epoch)
            writer.add_scalar("Time/validation", time.time() - t_start, epoch)
            early_stopping(validate_loss, redsdf)
            if early_stopping.early_stop:
                break

    # Test
    redsdf.eval()
    test_loss = 0
    n_points_test = len(test_loader.dataset)
    for points in test_loader:
        points = points.to(redsdf.device)
        norm_level_loss, J_nspace_proj_loss, cov_nspace_proj_loss, deviation_regularization = \
            redsdf.get_loss_components(points, poses)
        loss = norm_level_loss + J_nspace_proj_loss.mean() + cov_nspace_proj_loss.mean() + \
               gamma * deviation_regularization.mean()
        test_loss += loss.item() * points.shape[0]

    test_loss = test_loss / n_points_test
    writer.add_scalar("Loss/test", test_loss)
    writer.close()


if __name__ == '__main__':
    main()
