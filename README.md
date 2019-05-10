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

#### Datesets:ominiglot

description: each class just has 20 samples, most network trianing is overfitting in general GAN setting. the peformance of light model is better.

1. adDCGAN, channel=1, model-size=11M(D) + 14M(G)

conclusion: at first and sencond training epoch, generating new things (leanring), from epoch=3 to epoch=24, same mode, maybe the reanson is the small number of trianing images.

<div align="center">
<img src="/RUNs/Ominiglot_1/adDCGAN/image_map/fake_samples_epoch_000.png" height="200px" alt="fake_samples_epoch_000" >
<img src="/RUNs/Ominiglot_1/adDCGAN/image_map/real_samples_epoch_000.png" height="200px" alt="real_samples_epoch_000" >
</div>
<center>fake_samples_epoch_000 vs real_samples_epoch_000 </center>

<img src="/RUNs/Ominiglot_1/adDCGAN/image_map/fake_samples_epoch_001.png" height="200px" alt="fake_samples_epoch_001" >
<img src="/RUNs/Ominiglot_1/adDCGAN/image_map/real_samples_epoch_001.png" height="200px" alt="real_samples_epoch_001" >
</div>
<center>fake_samples_epoch_001 vs real_samples_epoch_001 </center>

<div align="center">
<img src="/RUNs/Ominiglot_1/adDCGAN/image_map/fake_samples_epoch_003.png" height="300px" alt="fake_samples_epoch_003" >
<img src="/RUNs/Ominiglot_1/adDCGAN/image_map/real_samples_epoch_003.png" height="300px" alt="real_samples_epoch_003" >
</div>
<center>fake_samples_epoch_003 vs real_samples_epoch_003 </center>

<div align="center">
<img src="/RUNs/Ominiglot_1/adDCGAN/image_map/fake_samples_epoch_024.png" height="300px" alt="fake_samples_epoch_024" >
<img src="/RUNs/Ominiglot_1/adDCGAN/image_map/real_samples_epoch_024.png" height="300px" alt="real_samples_epoch_024" >
</div>
<center>fake_samples_epoch_024 vs real_samples_epoch_024 </center>

GAN metric: 31 dimension (4 feature spaces * 7 scores + incep + modescore + fid), has problem




2. simpleDCGAN, channel=1, model-size=44M(D) + 50M(G)

conclusion: generating same mode at every epoch

<div align="center">
<img src="/RUNs/Ominiglot_1/simpleDCGAN/image_map/fake_samples_epoch_000.png" height="300px" alt="fake_samples_epoch_000" >
<img src="/RUNs/Ominiglot_1/simpleDCGAN/image_map/real_samples_epoch_000.png" height="300px" alt="real_samples_epoch_000" >
</div>
<center>fake_samples_epoch_000 vs real_samples_epoch_000 </center>

<div align="center">
<img src="/RUNs/Ominiglot_1/simpleDCGAN/image_map/fake_samples_epoch_047.png" height="300px" alt="fake_samples_epoch_006" >
<img src="/RUNs/Ominiglot_1/simpleDCGAN/image_map/real_samples_epoch_047.png" height="300px" alt="real_samples_epoch_006" >
</div>
<center>fake_samples_epoch_047 vs real_samples_epoch_047 </center>

<div align="center">
<img src="/RUNs/Ominiglot_1/simpleDCGAN/train_hist.png" height="300px" alt="train_hist" >
</div>
<center>train_hist </center>

GAN metric: 31 dimension (4 feature spaces * 7 scores + incep + modescore + fid), has problem

3. WGAN_GP_DCGAN, channel=1, model-size=77M(D) + 79M(G)

conclusion: generating new information in each epoch, but fail to learn correctly. 

<div align="center">
<img src="/RUNs/Ominiglot_1/WGAN_GP_DCGAN/image_map/fake_samples_epoch_000.png" height="200px" alt="fake_samples_epoch_000" >
<img src="/RUNs/Ominiglot_1/WGAN_GP_DCGAN/image_map/real_samples_epoch_000.png" height="200px" alt="real_samples_epoch_000" >
</div>
<center>fake_samples_epoch_000 vs real_samples_epoch_000 </center>

<img src="/RUNs/Ominiglot_1/WGAN_GP_DCGAN/image_map/fake_samples_epoch_010.png" height="200px" alt="fake_samples_epoch_010" >
<img src="/RUNs/Ominiglot_1/WGAN_GP_DCGAN/image_map/real_samples_epoch_010.png" height="200px" alt="real_samples_epoch_010" >
</div>
<center>fake_samples_epoch_010 vs real_samples_epoch_010 </center>

<div align="center">
<img src="/RUNs/Ominiglot_1/WGAN_GP_DCGAN/image_map/fake_samples_epoch_032.png" height="200px" alt="fake_samples_epoch_032" >
<img src="/RUNs/Ominiglot_1/WGAN_GP_DCGAN/image_map/real_samples_epoch_032.png" height="200px" alt="real_samples_epoch_032" >
</div>
<center>fake_samples_epoch_032 vs real_samples_epoch_032 </center>

<div align="center">
<img src="/RUNs/Ominiglot_1/WGAN_GP_DCGAN/image_map/fake_samples_epoch_063.png" height="200px" alt="fake_samples_epoch_063" >
<img src="/RUNs/Ominiglot_1/WGAN_GP_DCGAN/image_map/real_samples_epoch_063.png" height="200px" alt="real_samples_epoch_063" >
</div>
<center>fake_samples_epoch_063 vs real_samples_epoch_063 </center>

GAN metric: 31 dimension (4 feature spaces * 7 scores + incep + modescore + fid), has problem






#### Datesets:FIGR

description: some class has enough samples while the remaining classes just has several images, so different models can generate reasonable imags in general GAN setting.

1. simpleDCGAN, FIGR dataset, channel=3

<div align="center">
<img src="/RUNs/FIGR_3/simpleDCGAN/image_map/fake_samples_epoch_000.png" height="300px" alt="fake_samples_epoch_000" >
<img src="/RUNs/FIGR_3/simpleDCGAN/image_map/real_samples_epoch_000.png" height="300px" alt="real_samples_epoch_000" >
</div>
<center>fake_samples_epoch_000 vs real_samples_epoch_000 </center>

haven't finished running and haven't generated other measurement index, two large datasets


2. simpleDCGAN, FIGR dataset, channel=1

<div align="center">
<img src="/RUNs/FIGR_1/simpleDCGAN/image_map/fake_samples_epoch_000.png" height="300px" alt="fake_samples_epoch_000" >
<img src="/RUNs/FIGR_1/simpleDCGAN/image_map/real_samples_epoch_000.png" height="300px" alt="real_samples_epoch_000" >
</div>
<center>fake_samples_epoch_000 vs real_samples_epoch_000 </center>

haven't finished running and haven't generated other measurement index, two large datasets




3. WGAN_GP_DCGAN, FIGR dataset, channel=3

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





