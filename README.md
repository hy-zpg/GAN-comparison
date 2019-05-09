This repository is modified with [GAN collection](https://github.com/znxlwm/pytorch-generative-model-collections) and [GAN metrics](https://github.com/xuqiantong/GAN-Metrics)

### Files

1. GAN-Metrics refer to [GAN metrics](https://github.com/xuqiantong/
2. pytorch-generative-model-collections refer to [GAN collection](https://github.com/znxlwm/
3. RUNs: the results, currently including (two dataset [FIGR and Omniglot, four GAN model [adjustable DCGAN, simple DCGAN, WGAN-GP-DCGAN and WGAN-GP-ResNet], two mode images [three channels and one channels]])


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

2. Datasets Omniglot
python train.py --network='adDCGAN' --dataset='Omniglot' --niter=25 --ndc=1
python train.py --network='WGAN_GP_DCGAN' --dataset='Omniglot' --niter=25 --ndc=1
python train.py --network='WGAN_GP_ResNet' --dataset='Omniglot' --niter=25 --ndc=3 --batchSize=16




### Experimental results

1. WGAN_GP_DCGAN, Omniglot dataset, channel=1

<div align="center">
<img src="/RUNs/Ominiglot_1/WGAN_GP_DCGAN/image_map/fake_samples_epoch_006.png" height="300px" alt="fake_samples_epoch_006" >
<img src="/RUNs/Ominiglot_1/WGAN_GP_DCGAN/image_map/real_samples_epoch_006.png" height="300px" alt="real_samples_epoch_006" >
</div>
<center>fake_samples_epoch_006 vs real_samples_epoch_006 </center>

<div align="center">
<img src="/RUNs/Ominiglot_1/WGAN_GP_DCGAN/image_map/fake_samples_epoch_047.png" height="300px" alt="fake_samples_epoch_006" >
<img src="/RUNs/Ominiglot_1/WGAN_GP_DCGAN/image_map/real_samples_epoch_047.png" height="300px" alt="real_samples_epoch_006" >
</div>
<center>fake_samples_epoch_047 vs real_samples_epoch_047 </center>

<div align="center">
<img src="/RUNs/Ominiglot_1/WGAN_GP_DCGAN/train_hist.png" height="300px" alt="train_hist" >
</div>
<center>train_hist </center>

GAN metric: 31 dimension (4 feature spaces * 7 scores + incep + modescore + fid), has problem


2. simpleDCGAN, FIGR dataset, channel=3

<div align="center">
<img src="/RUNs/FIGR_3/simpleDCGAN/image_map/fake_samples_epoch_000.png" height="300px" alt="fake_samples_epoch_000" >
<img src="/RUNs/FIGR_3/simpleDCGAN/image_map/real_samples_epoch_000.png" height="300px" alt="real_samples_epoch_000" >
</div>
<center>fake_samples_epoch_000 vs real_samples_epoch_000 </center>

haven't finished running and haven't generated other measurement index, two large datasets


3. simpleDCGAN, FIGR dataset, channel=1

<div align="center">
<img src="/RUNs/FIGR_1/simpleDCGAN/image_map/fake_samples_epoch_000.png" height="300px" alt="fake_samples_epoch_000" >
<img src="/RUNs/FIGR_1/simpleDCGAN/image_map/real_samples_epoch_000.png" height="300px" alt="real_samples_epoch_000" >
</div>
<center>fake_samples_epoch_000 vs real_samples_epoch_000 </center>

haven't finished running and haven't generated other measurement index, two large datasets




4. WGAN_GP_DCGAN, FIGR dataset, channel=3

<div align="center">
<img src="/RUNs/FIGR_3/WGAN_GP_DCGAN/image_map/fake_samples_epoch_000.png" height="300px" alt="fake_samples_epoch_000" >
<img src="/RUNs/FIGR_3/WGAN_GP_DCGAN/image_map/real_samples_epoch_000.png" height="300px" alt="real_samples_epoch_000" >
</div>
<center>fake_samples_epoch_000 vs real_samples_epoch_000 </center>

haven't finished running and haven't generated other measurement index, two large datasets


4. WGAN_GP_DCGAN, FIGR dataset, channel=1

<div align="center">
<img src="/RUNs/FIGR_1/WGAN_GP_DCGAN/image_map/fake_samples_epoch_000.png" height="300px" alt="fake_samples_epoch_000" >
<img src="/RUNs/FIGR_1/WGAN_GP_DCGAN/image_map/real_samples_epoch_000.png" height="300px" alt="real_samples_epoch_000" >
</div>
<center>fake_samples_epoch_000 vs real_samples_epoch_000 </center>

haven't finished running and haven't generated other measurement index, two large datasets, 







### local running
python train.py --network='simpleDCGAN' --dataset='Omniglot' --niter=25 --batchSize=32 --ndc=1

python train.py --network='adDCGAN' --dataset='Omniglot' --niter=25 --batchSize=32 --ndc=1

python train.py --network='WGAN_GP_DCGAN' --dataset='Omniglot' --niter=25 --batchSize=32 --ndc=1

python train.py --network='WGAN_GP_ResNet' --dataset='Omniglot' --niter=25 --batchSize=32 --ndc=1


nohup python -u train.py --network='simpleDCGAN' --dataset='Omniglot' --niter=25 --batchSize=32 --ndc=1 > simpleDCGAN_Omniglot.log 2>&1 &

nohup python -u train.py --network='adDCGAN' --dataset='Omniglot' --niter=25 --batchSize=32 --ndc=1 > adDCGAN_Omniglot.log 2>&1 &

nohup python -u train.py --network='WGAN_GP_DCGAN' --dataset='Omniglot' --niter=25 --batchSize=32 --ndc=1 > WGAN_GP_DCGAN_Omniglot.log 2>&1 &

nohup python -u train.py --network='WGAN_GP_ResNet' --dataset='Omniglot' --niter=25 --batchSize=16 --ndc=1 > WGAN_GP_ResNet_Omniglot.log 2>&1 &


nohup python -u train.py --network='WGAN_GP_DCGAN' --dataset='FIGR' --niter=2 --batchSize=64 --ndc=1 > WGAN_GP_DCGAN_FIGR.log 2>&1 &





