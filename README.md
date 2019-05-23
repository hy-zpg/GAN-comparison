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

1. network:DCGAN, optimization:simple,  model-size=11M(D) + 14M(G)

conclusion: model collapse?

<div align="center">
<img src="/RUNs-old/Ominiglot_simple_adDCGAN/image_map/fake_samples_epoch_005.png" height="300px" alt="fake_samples_epoch_000" >
<img src="/RUNs-old/Ominiglot_simple_adDCGAN/image_map/real_samples_epoch_005.png" height="300px" alt="real_samples_epoch_000" >
</div>
<center>fake_samples_epoch_005 vs real_samples_epoch_005 </center>

<div align="center">
<img src="/RUNs-old/Ominiglot_simple_adDCGAN/image_map/fake_samples_epoch_009.png" height="300px" alt="fake_samples_epoch_000" >
<img src="/RUNs-old/Ominiglot_simple_adDCGAN/image_map/real_samples_epoch_009.png" height="300px" alt="real_samples_epoch_000" >
</div>
<center>fake_samples_epoch_009 vs real_samples_epoch_009 </center>

<div align="center">
<img src="/RUNs-old/Ominiglot_simple_adDCGAN/image_map/fake_samples_epoch_049.png" height="300px" alt="fake_samples_epoch_000" >
<img src="/RUNs-old/Ominiglot_simple_adDCGAN/image_map/real_samples_epoch_049.png" height="300px" alt="real_samples_epoch_000" >
</div>
<center>fake_samples_epoch_049 vs real_samples_epoch_049 </center>



2. network:ResNet optimization , model-size=77M(D) + 79M(G)

conclusion: generating new information in each epoch, is trainable. 

<div align="center">
<img src="/RUNs-old-19/Ominiglot_WGAN_GP_ResNet/image_map/fake_samples_epoch_010.png" height="200px" alt="fake_samples_epoch_010" >
<img src="/RUNs-old-19/Ominiglot_WGAN_GP_ResNet/image_map/real_samples_epoch_010.png" height="200px" alt="real_samples_epoch_010" >
</div>
<center>fake_samples_epoch_010 vs real_samples_epoch_010 </center>


<div align="center">
<img src="/RUNs-old-19/Ominiglot_WGAN_GP_ResNet/image_map/fake_samples_epoch_017.png" height="200px" alt="fake_samples_epoch_017" >
<img src="/RUNs-old-19/Ominiglot_WGAN_GP_ResNet/image_map/real_samples_epoch_017.png" height="200px" alt="real_samples_epoch_017" >
</div>
<center>fake_samples_epoch_017 vs real_samples_epoch_017 </center>

<div align="center">
<img src="/RUNs-old-19/Ominiglot_WGAN_GP_ResNet/image_map/fake_samples_epoch_030.png" height="200px" alt="fake_samples_epoch_017" >
<img src="/RUNs-old-19/Ominiglot_WGAN_GP_ResNet/image_map/real_samples_epoch_030.png" height="200px" alt="real_samples_epoch_017" >
</div>
<center>fake_samples_epoch_030 vs real_samples_epoch_030 </center>



#### Datesets:small-FIGR
1. network:DCGAN, optimization:simple,  model-size=11M(D) + 14M(G)

<div align="center">
<img src="/RUNs-old/small_FIGR_simple_adDCGAN/image_map/fake_samples_epoch_010.png" height="200px" alt="fake_samples_epoch_017" >
<img src="/RUNs-old/small_FIGR_simple_adDCGAN/image_map/real_samples_epoch_010.png" height="200px" alt="real_samples_epoch_017" >
</div>
<center>fake_samples_epoch_010 vs real_samples_epoch_010 </center>

<div align="center">
<img src="/RUNs-old/small_FIGR_simple_adDCGAN/image_map/fake_samples_epoch_030.png" height="200px" alt="fake_samples_epoch_017" >
<img src="/RUNs-old/small_FIGR_simple_adDCGAN/image_map/real_samples_epoch_030.png" height="200px" alt="real_samples_epoch_017" >
</div>
<center>fake_samples_epoch_030 vs real_samples_epoch_030 </center>


<div align="center">
<img src="/RUNs-old/small_FIGR_simple_adDCGAN/image_map/fake_samples_epoch_063.png" height="200px" alt="fake_samples_epoch_017" >
<img src="/RUNs-old/small_FIGR_simple_adDCGAN/image_map/real_samples_epoch_063.png" height="200px" alt="real_samples_epoch_017" >
</div>
<center>fake_samples_epoch_063 vs real_samples_epoch_063 </center>



2. network:ResNet optimization , model-size=77M(D) + 79M(G)

<div align="center">
<img src="/RUNs-old-19/small_FIGR_WGAN_GP_ResNet/image_map/fake_samples_epoch_000.png" height="200px" alt="fake_samples_epoch_017" >
<img src="/RUNs-old-19/small_FIGR_WGAN_GP_ResNet/image_map/real_samples_epoch_000.png" height="200px" alt="real_samples_epoch_017" >
</div>
<center>fake_samples_epoch_000 vs real_samples_epoch_000 </center>

<div align="center">
<img src="/RUNs-old-19/small_FIGR_WGAN_GP_ResNet/image_map/fake_samples_epoch_060.png" height="200px" alt="fake_samples_epoch_017" >
<img src="/RUNs-old-19/small_FIGR_WGAN_GP_ResNet/image_map/real_samples_epoch_060.png" height="200px" alt="real_samples_epoch_017" >
</div>
<center>fake_samples_epoch_060 vs real_samples_epoch_060 </center>

<div align="center">
<img src="/RUNs-old-19/small_FIGR_WGAN_GP_ResNet/image_map/fake_samples_epoch_105.png" height="200px" alt="fake_samples_epoch_017" >
<img src="/RUNs-old-19/small_FIGR_WGAN_GP_ResNet/image_map/real_samples_epoch_105.png" height="200px" alt="real_samples_epoch_017" >
</div>
<center>fake_samples_epoch_105 vs real_samples_epoch_105 </center>


<div align="center">
<img src="/RUNs-old-19/small_FIGR_WGAN_GP_ResNet/image_map/fake_samples_epoch_306.png" height="200px" alt="fake_samples_epoch_017" >
<img src="/RUNs-old-19/small_FIGR_WGAN_GP_ResNet/image_map/real_samples_epoch_306.png" height="200px" alt="real_samples_epoch_017" >
</div>
<center>fake_samples_epoch_306 vs real_samples_epoch_306 </center>


<div align="center">
<img src="/RUNs-old-19/small_FIGR_WGAN_GP_ResNet/image_map/fake_samples_epoch_777.png" height="200px" alt="fake_samples_epoch_017" >
<img src="/RUNs-old-19/small_FIGR_WGAN_GP_ResNet/image_map/real_samples_epoch_777.png" height="200px" alt="real_samples_epoch_017" >
</div>
<center>fake_samples_epoch_777 vs real_samples_epoch_777 </center>





