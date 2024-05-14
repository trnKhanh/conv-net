#!/bin/bash

python main.py --fig-dir figs --train --dataset MNIST --hidden-channels 100 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/mnist_net_100.pt --save-freq 10 \
  --save-best-path models/mnist_net_100_best.pt --device cpu

python main.py --fig-dir figs --train --dataset MNIST --hidden-channels 100 500 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/mnist_net_100_500.pt --save-freq 10 \
  --save-best-path models/mnist_net_100_500_best.pt --device cpu

python main.py --fig-dir figs --train --dataset MNIST --hidden-channels 100 500 700 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/mnist_net_100_500_700.pt --save-freq 10 \
  --save-best-path models/mnist_net_100_500_700_best.pt --device cpu

python main.py --fig-dir figs --train --dataset MNIST --hidden-channels 100 500 700 900 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/mnist_net_100_500_700_900.pt --save-freq 10 \
  --save-best-path models/mnist_net_100_500_700_900_best.pt --device cpu

python main.py --fig-dir figs --train --dataset MNIST --hidden-channels 100 500 700 900 1000 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/mnist_net_100_500_700_900_1000.pt --save-freq 10 \
  --save-best-path models/mnist_net_100_500_700_900_1000_best.pt --device cpu

python main.py --fig-dir figs --train --dataset MNIST --hidden-channels 200 1000 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/mnist_net_200_1000.pt --save-freq 10 \
  --save-best-path models/mnist_net_200_1000_best.pt --device cpu

python main.py --fig-dir figs --train --dataset MNIST --hidden-channels 400 2000 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/mnist_net_400_2000.pt --save-freq 10 \
  --save-best-path models/mnist_net_400_2000_best.pt --device cpu

# FashinMNIST
python main.py --fig-dir figs --train --dataset FashionMNIST --hidden-channels 100 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/fmnist_net_100.pt --save-freq 10 \
  --save-best-path models/fmnist_net_100_best.pt --device cpu

python main.py --fig-dir figs --train --dataset FashionMNIST --hidden-channels 100 500 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/fmnist_net_100_500.pt --save-freq 10 \
  --save-best-path models/fmnist_net_100_500_best.pt --device cpu

python main.py --fig-dir figs --train --dataset FashionMNIST --hidden-channels 100 500 700 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/fmnist_net_100_500_700.pt --save-freq 10 \
  --save-best-path models/fmnist_net_100_500_700_best.pt --device cpu

python main.py --fig-dir figs --train --dataset FashionMNIST --hidden-channels 100 500 700 900 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/fmnist_net_100_500_700_900.pt --save-freq 10 \
  --save-best-path models/fmnist_net_100_500_700_900_best.pt --device cpu

python main.py --fig-dir figs --train --dataset FashionMNIST --hidden-channels 100 500 700 900 1000 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/fmnist_net_100_500_700_900_1000.pt --save-freq 10 \
  --save-best-path models/fmnist_net_100_500_700_900_1000_best.pt --device cpu

python main.py --fig-dir figs --train --dataset FashionMNIST --hidden-channels 200 1000 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/fmnist_net_200_1000.pt --save-freq 10 \
  --save-best-path models/fmnist_net_200_1000_best.pt --device cpu

python main.py --fig-dir figs --train --dataset FashionMNIST --hidden-channels 400 2000 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/fmnist_net_400_2000.pt --save-freq 10 \
  --save-best-path models/fmnist_net_400_2000_best.pt --device cpu

python main.py --hog --fig-dir figs --train --dataset MNIST --hidden-channels 100 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/hog_mnist_net_100.pt --save-freq 10 \
  --save-best-path models/hog_mnist_net_100_best.pt --device cpu

python main.py --hog --fig-dir figs --train --dataset MNIST --hidden-channels 100 500 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/hog_mnist_net_100_500.pt --save-freq 10 \
  --save-best-path models/hog_mnist_net_100_500_best.pt --device cpu

python main.py --hog --fig-dir figs --train --dataset MNIST --hidden-channels 100 500 700 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/hog_mnist_net_100_500_700.pt --save-freq 10 \
  --save-best-path models/hog_mnist_net_100_500_700_best.pt --device cpu

python main.py --hog --fig-dir figs --train --dataset MNIST --hidden-channels 100 500 700 900 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/hog_mnist_net_100_500_700_900.pt --save-freq 10 \
  --save-best-path models/hog_mnist_net_100_500_700_900_best.pt --device cpu

python main.py --hog --fig-dir figs --train --dataset MNIST --hidden-channels 100 500 700 900 1000 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/hog_mnist_net_100_500_700_900_1000.pt --save-freq 10 \
  --save-best-path models/hog_mnist_net_100_500_700_900_1000_best.pt --device cpu

python main.py --hog --fig-dir figs --train --dataset MNIST --hidden-channels 200 1000 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/hog_mnist_net_200_1000.pt --save-freq 10 \
  --save-best-path models/hog_mnist_net_200_1000_best.pt --device cpu

python main.py --hog --fig-dir figs --train --dataset MNIST --hidden-channels 400 2000 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/hog_mnist_net_400_2000.pt --save-freq 10 \
  --save-best-path models/hog_mnist_net_400_2000_best.pt --device cpu

# FashinMNIST
python main.py --hog --fig-dir figs --train --dataset FashionMNIST --hidden-channels 100 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/hog_fmnist_net_100.pt --save-freq 10 \
  --save-best-path models/hog_fmnist_net_100_best.pt --device cpu

python main.py --hog --fig-dir figs --train --dataset FashionMNIST --hidden-channels 100 500 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/hog_fmnist_net_100_500.pt --save-freq 10 \
  --save-best-path models/hog_fmnist_net_100_500_best.pt --device cpu

python main.py --hog --fig-dir figs --train --dataset FashionMNIST --hidden-channels 100 500 700 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/hog_fmnist_net_100_500_700.pt --save-freq 10 \
  --save-best-path models/hog_fmnist_net_100_500_700_best.pt --device cpu

python main.py --hog --fig-dir figs --train --dataset FashionMNIST --hidden-channels 100 500 700 900 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/hog_fmnist_net_100_500_700_900.pt --save-freq 10 \
  --save-best-path models/hog_fmnist_net_100_500_700_900_best.pt --device cpu

python main.py --hog --fig-dir figs --train --dataset FashionMNIST --hidden-channels 100 500 700 900 1000 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/hog_fmnist_net_100_500_700_900_1000.pt --save-freq 10 \
  --save-best-path models/hog_fmnist_net_100_500_700_900_1000_best.pt --device cpu

python main.py --hog --fig-dir figs --train --dataset FashionMNIST --hidden-channels 200 1000 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/hog_fmnist_net_200_1000.pt --save-freq 10 \
  --save-best-path models/hog_fmnist_net_200_1000_best.pt --device cpu

python main.py --hog --fig-dir figs --train --dataset FashionMNIST --hidden-channels 400 2000 --num-classes 10 \
  --epochs 100 --batch-size 512 --learning-rate 0.001 \
  --save-path models/hog_fmnist_net_400_2000.pt --save-freq 10 \
  --save-best-path models/hog_fmnist_net_400_2000_best.pt --device cpu


