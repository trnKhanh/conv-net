#!/bin/bash

# python main.py --dataset MNIST --hidden-channels 100 --num-classes 10 \
#   --load-ckpt models/mnist_net_100_best.pt --device cpu
#
# python main.py --dataset MNIST --hidden-channels 100 500 --num-classes 10 \
#   --load-ckpt models/mnist_net_100_500_best.pt --device cpu
#
# python main.py --dataset MNIST --hidden-channels 100 500 700 --num-classes 10 \
#   --load-ckpt models/mnist_net_100_500_700_best.pt --device cpu
#
# python main.py --dataset MNIST --hidden-channels 100 500 700 900 --num-classes 10 \
#   --load-ckpt models/mnist_net_100_500_700_900_best.pt --device cpu
#
# python main.py --dataset MNIST --hidden-channels 100 500 700 900 1000 --num-classes 10 \
#   --load-ckpt models/mnist_net_100_500_700_900_1000_best.pt --device cpu
#
# python main.py --dataset MNIST --hidden-channels 200 1000 --num-classes 10 \
#   --load-ckpt models/mnist_net_200_1000_best.pt --device cpu
#
# python main.py --dataset MNIST --hidden-channels 400 2000 --num-classes 10 \
#   --load-ckpt models/mnist_net_400_2000_best.pt --device cpu

# FashinMNIST
python main.py --dataset FashionMNIST --hidden-channels 100 --num-classes 10 \
  --load-ckpt models/fmnist_net_100_best.pt --device cpu

python main.py --dataset FashionMNIST --hidden-channels 100 500 --num-classes 10 \
  --load-ckpt models/fmnist_net_100_500_best.pt --device cpu

python main.py --dataset FashionMNIST --hidden-channels 100 500 700 --num-classes 10 \
  --load-ckpt models/fmnist_net_100_500_700_best.pt --device cpu

python main.py --dataset FashionMNIST --hidden-channels 100 500 700 900 --num-classes 10 \
  --load-ckpt models/fmnist_net_100_500_700_900_best.pt --device cpu

python main.py --dataset FashionMNIST --hidden-channels 100 500 700 900 1000 --num-classes 10 \
  --load-ckpt models/fmnist_net_100_500_700_900_1000_best.pt --device cpu

python main.py --dataset FashionMNIST --hidden-channels 200 1000 --num-classes 10 \
  --load-ckpt models/fmnist_net_200_1000_best.pt --device cpu

python main.py --dataset FashionMNIST --hidden-channels 400 2000 --num-classes 10 \
  --load-ckpt models/fmnist_net_400_2000_best.pt --device cpu

# python main.py --hog --dataset MNIST --hidden-channels 100 --num-classes 10 \
#   --load-ckpt models/hog_mnist_net_100_best.pt --device cpu
#
# python main.py --hog --dataset MNIST --hidden-channels 100 500 --num-classes 10 \
#   --load-ckpt models/hog_mnist_net_100_500_best.pt --device cpu
#
# python main.py --hog --dataset MNIST --hidden-channels 100 500 700 --num-classes 10 \
#   --load-ckpt models/hog_mnist_net_100_500_700_best.pt --device cpu
#
# python main.py --hog --dataset MNIST --hidden-channels 100 500 700 900 --num-classes 10 \
#   --load-ckpt models/hog_mnist_net_100_500_700_900_best.pt --device cpu
#
# python main.py --hog --dataset MNIST --hidden-channels 100 500 700 900 1000 --num-classes 10 \
#   --load-ckpt models/hog_mnist_net_100_500_700_900_1000_best.pt --device cpu
#
# python main.py --hog --dataset MNIST --hidden-channels 200 1000 --num-classes 10 \
#   --load-ckpt models/hog_mnist_net_200_1000_best.pt --device cpu
#
# python main.py --hog --dataset MNIST --hidden-channels 400 2000 --num-classes 10 \
#   --load-ckpt models/hog_mnist_net_400_2000_best.pt --device cpu
#
# # FashinMNIST
# python main.py --hog --dataset FashionMNIST --hidden-channels 100 --num-classes 10 \
#   --load-ckpt models/hog_fmnist_net_100_best.pt --device cpu
#
# python main.py --hog --dataset FashionMNIST --hidden-channels 100 500 --num-classes 10 \
#   --load-ckpt models/hog_fmnist_net_100_500_best.pt --device cpu
#
# python main.py --hog --dataset FashionMNIST --hidden-channels 100 500 700 --num-classes 10 \
#   --load-ckpt models/hog_fmnist_net_100_500_700_best.pt --device cpu
#
# python main.py --hog --dataset FashionMNIST --hidden-channels 100 500 700 900 --num-classes 10 \
#   --load-ckpt models/hog_fmnist_net_100_500_700_900_best.pt --device cpu
#
# python main.py --hog --dataset FashionMNIST --hidden-channels 100 500 700 900 1000 --num-classes 10 \
#   --load-ckpt models/hog_fmnist_net_100_500_700_900_1000_best.pt --device cpu
#
# python main.py --hog --dataset FashionMNIST --hidden-channels 200 1000 --num-classes 10 \
#   --load-ckpt models/hog_fmnist_net_200_1000_best.pt --device cpu
#
# python main.py --hog --dataset FashionMNIST --hidden-channels 400 2000 --num-classes 10 \
#   --load-ckpt models/hog_fmnist_net_400_2000_best.pt --device cpu
#
#
