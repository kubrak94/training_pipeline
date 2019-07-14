import argparse

from src.train.train_helper import run_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--hparams', type=str, required=True,
        help='path to .yaml file with config for training'
    )
    
    parser.add_argument(
        '--resume', default='', type=str, metavar='PATH',
        help='path to the latest checkpoint (default: none)'
    )
    
    args = parser.parse_args()

    run_train(args)
