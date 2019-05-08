This repository is modified with [GAN collection](https://github.com/znxlwm/pytorch-generative-model-collections) and [GAN metrics](https://github.com/xuqiantong/GAN-Metrics)

### Files

1. GAN-Metrics refer to [GAN metrics](https://github.com/xuqiantong/
2. pytorch-generative-model-collections refer to [GAN collection](https://github.com/znxlwm/
3. RUNs: the results, currently including (two dataset [FIGR and Ominiglot, four GAN model [adjustable DCGAN, simple DCGAN, WGAN-GP-DCGAN and WGAN-GP-ResNet], two mode images [three channels and one channels]])


### Codes
1. Discriminator.py: different Discriminator models
2. Generator.py: different Generator models
3. data_utils: preparing data from directory, two modes including grey images and color images
4. metric.py: different GAN metric including 4 feature spaces and 7 measurements index
5.visualization.py: visulization for different GAN models
6.GAN_environments.py: core code, different GANs training
7.train.py: setting important parameters, such as selected datasets, selected GAN model and other training parameters.


### Scripts
1. Datasets FIGR
python train.py --network='adDCGAN' --dataset='FIGR' --niter=25 --ndc=3 
python train.py --network='adDCGAN' --dataset='FIGR' --niter=25 --ndc=1
python train.py --network='WGAN_GP_DCGAN' --dataset='FIGR' --niter=25 --ndc=1
python train.py --network='WGAN_GP_ResNet' --dataset='FIGR' --niter=25 --ndc=1
python train.py --network='simpleDCGAN' --dataset='FIGR' --niter=25 --ndc=1

2. Datasets Ominiglot
python train.py --network='adDCGAN' --dataset='Ominiglot' --niter=25 --ndc=1
python train.py --network='WGAN_GP_DCGAN' --dataset='Ominiglot' --niter=25 --ndc=1
python train.py --network='WGAN_GP_ResNet' --dataset='Ominiglot' --niter=25 --ndc=3 --batchSize=16


### local running
python train.py --network='simpleDCGAN' --dataset='Ominiglot' --niter=25 --batchSize=32 --ndc=1
python train.py --network='adDCGAN' --dataset='Ominiglot' --niter=25 --batchSize=32 --ndc=1
python train.py --network='WGAN_GP_DCGAN' --dataset='Ominiglot' --niter=25 --batchSize=32 --ndc=1
python train.py --network='WGAN_GP_ResNet' --dataset='Ominiglot' --niter=25 --batchSize=32 --ndc=1


nohup python -u train.py --network='simpleDCGAN' --dataset='Ominiglot' --niter=25 --batchSize=32 --ndc=1 > simpleDCGAN_Ominiglot.log 2>&1 &
nohup python -u train.py --network='adDCGAN' --dataset='Ominiglot' --niter=25 --batchSize=32 --ndc=1 > adDCGAN_Ominiglot.log 2>&1 &
nohup python -u train.py --network='WGAN_GP_DCGAN' --dataset='Ominiglot' --niter=25 --batchSize=32 --ndc=1 > WGAN_GP_DCGAN_Ominiglot.log 2>&1 &
nohup python -u train.py --network='WGAN_GP_ResNet' --dataset='Ominiglot' --niter=25 --batchSize=16 --ndc=1 > WGAN_GP_ResNet_Ominiglot.log 2>&1 &


nohup python -u train.py --network='WGAN_GP_DCGAN' --dataset='FIGR' --niter=2 --batchSize=64 --ndc=1 > WGAN_GP_DCGAN_FIGR.log 2>&1 &


