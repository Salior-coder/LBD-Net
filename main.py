import argparse
import os
from torch.backends import cudnn
from dataSet import get_loader
from train_test import Solver

def main(config):
    cudnn.benchmark = True

    if not os.path.exists(config.model_path):
        os.mkdir(config.model_path)
    if not os.path.exists(config.result_path):
        os.mkdir(config.result_path)

    train_loader = get_loader(config.data_path, 'train', config.batch_size, config.num_workers)
    test_loader = get_loader(config.data_path, 'test', 1, config.num_workers)
    print('data load success')
    print('train dataset len %d' % (len(train_loader) * config.batch_size))
    print('test dataset len %d' % len(test_loader))

    results_path = config.result_path
    layer_results_path = os.path.join(results_path, 'layers1')
    dst_results_path = os.path.join(results_path, 'dst_layer')

    if not os.path.exists(results_path):
        os.mkdir(results_path)
    if not os.path.exists(layer_results_path):
        os.mkdir(layer_results_path)
        # os.mkdir(dst_results_path)

    solver = Solver(config, train_loader, test_loader)
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test(layer_results_path, dst_results_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default='')
    parser.add_argument('--cuda-idx', type=int, default=1)
    parser.add_argument('--mode', type=str, default='')  # [train/test]
    parser.add_argument('--img-ch', type=int, default='')
    parser.add_argument('--num-classes', type=int, default='')
    parser.add_argument('--result-path', type=str, default='')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--pretrained-model', type=str, default='')

    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)

    config = parser.parse_args()

    main(config)
