from Discriminator import DCGANdiscriminator, adDCGANDiscriminator, ResNetDiscriminator, DCGANDiscriminator
from Generator import DCGANgenerator, adDCGANGenerator, ResNetGenerator, DCGANGenerator
import torch

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import time
from torch import autograd
from torch.autograd import Variable
from visualization import DCGAN_show_result,WGAN_show_result,show_train_hist
from tensorboardX import SummaryWriter
import pickle


def adDCGAN(opt,metric,train_loader,dataroot,outf):
	def weights_init(m):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
			m.weight.data.normal_(0.0, 0.02)
		elif classname.find('BatchNorm') != -1:
			m.weight.data.normal_(1.0, 0.02)
			m.bias.data.fill_(0)

	#### Models building ####
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# print('here',device)
	netG = adDCGANGenerator(opt.ngpu,opt.nz,opt.ndc,opt.ngf).to(device)
	netG.apply(weights_init)
	if opt.netG != '':
		netG.load_state_dict(torch.load(opt.netG))
	netD = adDCGANDiscriminator(opt.ngpu,opt.nz,opt.ndc,opt.ndf).to(device)
	netD.apply(weights_init)
	if opt.netD != '':
		netD.load_state_dict(torch.load(opt.netD))

	# netG=nn.DataParallel(netG,device_ids=[0,1])
	# netD=nn.DataParallel(netD,device_ids=[0,1])


	criterion = nn.BCELoss()
	fixed_noise = torch.randn(opt.batchSize, opt.nz, 1, 1, device=device)
	real_label = 1
	fake_label = 0
	# setup optimizer
	optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
	optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
	
	if opt.netD != '':
		# [emd-mmd-knn(knn,real,fake,precision,recall)]*4 - IS - mode_score - FID
		score_tr = np.zeros((opt.niter, 4*7+3))
		# compute initial score
		s = metric.compute_score_raw(opt.dataset, opt.imageSize,dataroot, opt.sampleSize, 16, outf+'/real/', outf+'/fake/',
									 netG, opt.nz, conv_model='inception_v3', workers=int(opt.workers))
		score_tr[0] = s
		np.save('%s/score_tr.npy' % (outf), score_tr)

	#### Models training ####
	for epoch in range(opt.niter):
		for i, data in enumerate(train_loader, 0):
			# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
			# train with real
			netD.zero_grad()
			real_cpu = data[0].to(device,dtype=torch.float)
			batch_size = real_cpu.size(0)
			label = torch.full((batch_size,), real_label, device=device)

			output = netD(real_cpu)
			errD_real = criterion(output, label)
			errD_real.backward()
			D_x = output.mean().item()

			# train with fake
			noise = torch.randn(batch_size, opt.nz, 1, 1, device=device)
			fake = netG(noise)
			label.fill_(fake_label)
			output = netD(fake.detach())
			errD_fake = criterion(output, label)
			errD_fake.backward()
			D_G_z1 = output.mean().item()
			errD = errD_real + errD_fake
			optimizerD.step()

			# (2) Update G network: maximize log(D(G(z)))
			netG.zero_grad()
			label.fill_(real_label)  # fake labels are real for generator cost
			output = netD(fake)
			errG = criterion(output, label)
			errG.backward()
			D_G_z2 = output.mean().item()
			optimizerG.step()

			if i % 10 == 0:
				print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
					  % (epoch, opt.niter, i, len(train_loader),
						 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
			if i % 100 == 0:
				vutils.save_image(real_cpu,
						'%s/real_samples.png' % outf,
						normalize=True)
				fake = netG(fixed_noise)
				vutils.save_image(fake.detach(),
						'%s/fake_samples_epoch_%03d.png' % (outf, epoch),
						normalize=True)

		# do checkpointing
		torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
		torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))

		#### metric scores computing (key function) ####
		score_tr = np.zeros((opt.niter, 4*7+3))
		s = metric.compute_score_raw(opt.dataset, opt.imageSize, dataroot, opt.sampleSize, opt.batchSize, outf+'/real/', outf+'/fake/',\
									 netG, opt.nz, conv_model='inception_v3', workers=int(opt.workers))
		score_tr[epoch] = s

	# save final metric scores of all epoches
	np.save('%s/score_tr_ep.npy' % outf, score_tr)
	print('##### training completed :) #####')
	print('### metric scores output is scored at %s/score_tr_ep.npy ###' % outf)



def simpleDCGAN(opt,metric,train_loader,dataroot,outf):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	fixed_noise = torch.randn(opt.batchSize, opt.nz, 1, 1, device=device)
	G = DCGANgenerator(128,opt.ndc)
	D = DCGANdiscriminator(128,opt.ndc)
	print(G)
	print(D)
	
	G.weight_init(mean=0.0, std=0.02)
	D.weight_init(mean=0.0, std=0.02)
	if opt.netG != '':
		G.load_state_dict(torch.load(opt.netG))
	if opt.netD != '':
		D.load_state_dict(torch.load(opt.netD))

	G.cuda()
	D.cuda()
	G=nn.DataParallel(G,device_ids=[0,1])
	D=nn.DataParallel(D,device_ids=[0,1])

	BCE_loss = nn.BCELoss()

	G_optimizer = optim.Adam(G.parameters(), lr=opt.lr, betas=(0.5, 0.999))
	D_optimizer = optim.Adam(D.parameters(), lr=opt.lr, betas=(0.5, 0.999))

	if opt.netD != '':
		score_tr = np.zeros((opt.niter, 4*7+3))
		s = metric.compute_score_raw(opt.dataset, opt.imageSize, dataroot, opt.sampleSize, opt.batchSize, outf+'/real/', outf+'/fake/',\
											 G, opt.nz, conv_model='inception_v3', workers=int(opt.workers))
		score_tr[0] = s
		np.save('%s/score_tr.npy' % (outf), score_tr)


	train_hist = {}
	train_hist['D_losses'] = []
	train_hist['G_losses'] = []
	train_hist['per_epoch_ptimes'] = []
	train_hist['total_ptime'] = []

	print('Training start!')
	start_time = time.time()
	for epoch in range(opt.niter):
		print('this is epoch {}'.format(epoch))
		D_losses = []
		G_losses = []

		# learning rate decay
		if (epoch+1) == 11:
			G_optimizer.param_groups[0]['lr'] /= 10
			D_optimizer.param_groups[0]['lr'] /= 10
			print("learning rate change!")

		if (epoch+1) == 16:
			G_optimizer.param_groups[0]['lr'] /= 10
			D_optimizer.param_groups[0]['lr'] /= 10
			print("learning rate change!")

		### DCGAN
		### Discriminator: can make accurate distinguish, D_train_loss = fake_loss(D(G(z_)),y_fake_) + real_loss(D(x_),y_real_), training discriminator first
		### Generator: can generate the same fake image as real image, G_train_loss = (D(G(z_),y_real_), then training generator
		print('starting training')
		num_iter = 0
		epoch_start_time = time.time()
		print(len(train_loader))

		for i, data in enumerate(train_loader, 0):
		# for x_, _ in train_loader:
			# train discriminator D
			D.zero_grad()
			mini_batch=data[0].size()[0]
			
			# if isCrop:
			# 	x_ = x_[:, :, 22:86, 22:86]

			y_real_ = torch.ones(mini_batch)
			y_fake_ = torch.zeros(mini_batch)

			x, y_real_, y_fake_ = Variable(data[0].cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
			# [128, 3, 64, 64]
			# x_=x
			x_ = x.to(device,dtype=torch.float)

			D_result = D(x_).squeeze()
			D_x = D_result.mean().item()
			# [128]
			D_real_loss = BCE_loss(D_result, y_real_)

			z_ = torch.randn((mini_batch, opt.nz)).view(-1, opt.nz, 1, 1)
			z_ = Variable(z_.cuda())
			G_result = G(z_)
			# [128, 3, 64, 64]

			D_result = D(G_result).squeeze()
			D_G_z1 = D_result.mean().item()
			D_fake_loss = BCE_loss(D_result, y_fake_)
			D_fake_score = D_result.data.mean()

			D_train_loss = D_real_loss + D_fake_loss

			D_train_loss.backward()
			D_optimizer.step()

			D_losses.append(D_train_loss.data)

			# train generator G
			G.zero_grad()

			z_ = torch.randn((mini_batch, opt.nz)).view(-1, opt.nz, 1, 1)
			z_ = Variable(z_.cuda())

			G_result = G(z_)
			D_result = D(G_result).squeeze()
			D_G_z2 = D_result.mean().item()
			G_train_loss = BCE_loss(D_result, y_real_)
			G_train_loss.backward()
			G_optimizer.step()

			G_losses.append(G_train_loss.data)

			num_iter += 1

			if i % 10 == 0:
				print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
					  % (epoch, opt.niter, i, len(train_loader),
						 D_train_loss.item(), G_train_loss.item(), D_x, D_G_z1, D_G_z2))
			if i % 100 == 0:
				vutils.save_image(x_,
						'%s/real_samples.png' % outf,
						normalize=True)
				fake = G(fixed_noise)
				vutils.save_image(fake.detach(),
						'%s/fake_samples_epoch_%03d.png' % (outf, epoch),
						normalize=True)

		# do checkpointing
		torch.save(G.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
		torch.save(D.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))

		#### metric scores computing (key function) ####
		score_tr = np.zeros((opt.niter, 4*7+3))
		s = metric.compute_score_raw(opt.dataset, opt.imageSize, dataroot, opt.sampleSize, opt.batchSize, outf+'/real/', outf+'/fake/',\
									 G, opt.nz, conv_model='inception_v3', workers=int(opt.workers))
		score_tr[epoch] = s

	




		epoch_end_time = time.time()
		per_epoch_ptime = epoch_end_time - epoch_start_time


		print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), opt.niter, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
																  torch.mean(torch.FloatTensor(G_losses))))
		
		p = outf + str(epoch + 1) + '.png'
		DCGAN_show_result((epoch+1),G, p, 5, opt.nz)
		train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
		train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
		train_hist['per_epoch_ptimes'].append(per_epoch_ptime)



	end_time = time.time()
	total_ptime = end_time - start_time
	train_hist['total_ptime'].append(total_ptime)

	print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), opt.niter, total_ptime))
	print("Training finish!... save training results")
	torch.save(G.state_dict(), '%s/G_epoch_%d.pth' % (outf, epoch))
	torch.save(D.state_dict(), '%s/D_epoch_%d.pth' % (outf, epoch))


	with open(outf+'/train_hist.pkl', 'wb') as f:
		pickle.dump(train_hist, f)

	show_train_hist(train_hist, save=True, path=outf+'/train_hist.png')

	images = []
	for e in range(opt.niter):
		img_name = outf + str(e + 1) + '.png'
		images.append(imageio.imread(img_name))
	imageio.mimsave(outf+'/generation_animation.gif', images, fps=5)

	# save final metric scores of all epoches
	np.save('%s/score_tr_ep.npy' % outf, score_tr)
	print('##### training completed :) #####')
	print('### metric scores output is scored at %s/score_tr_ep.npy ###' % outf)


def WGAN_GP(opt,metric,train_loader,dataroot,outf,neural_network):
	def normalize_data(data):
	    data *= 2
	    data -= 1
	    return data


	def unnormalize_data(data):
	    data += 1
	    data /= 2
	    return data

	def wassertein_loss(inputs, targets):
	    return torch.mean(inputs * targets)


	def calc_gradient_penalty(discriminator, real_batch, fake_batch):
	    epsilon = torch.rand(real_batch.shape[0], 1, device=device)
	    interpolates = epsilon.view(-1, 1, 1, 1) * real_batch + (1 - epsilon).view(-1, 1, 1, 1) * fake_batch
	    interpolates = autograd.Variable(interpolates, requires_grad=True)
	    disc_interpolates = discriminator(interpolates)

	    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
	                              grad_outputs=torch.ones(disc_interpolates.size(), device=device),
	                              create_graph=True, retain_graph=True, only_inputs=True)[0]

	    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
	    return gradient_penalty


	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	fixed_noise = torch.randn(opt.batchSize, opt.nz, 1, 1, device=device)
	D = eval(neural_network + 'Discriminator(opt.ndc, opt.imageSize, opt.imageSize)').to(device)
	G = eval(neural_network + 'Generator(opt.nz, opt.ndc, opt.imageSize, opt.imageSize)').to(device)
	# print(G)
	# print(D)
	G.cuda()
	D.cuda()

	print(D)
	print(G)

	if opt.netG != '':
		G.load_state_dict(torch.load(opt.netG))
	if opt.netD != '':
		D.load_state_dict(torch.load(opt.netD))


	G=nn.DataParallel(G,device_ids=[0,1])
	D=nn.DataParallel(D,device_ids=[0,1])


	## discriminator: [real_image_label(1),fake_image_label(-1)] -> loss
	## generator: [real_image_label(1)]

	D_optim = optim.SGD(params=D.parameters(), lr=opt.lr)
	G_optim = optim.SGD(params=G.parameters(), lr=opt.lr)


	if opt.netD != '':
		score_tr = np.zeros((opt.niter, 4*7+3))
		s = metric.compute_score_raw(opt.dataset, opt.imageSize, dataroot, opt.sampleSize, opt.batchSize, outf+'/real/', outf+'/fake/',\
													 G, opt.nz, conv_model='inception_v3', workers=int(opt.workers))
		score_tr[0] = s
		np.save('%s/score_tr.npy' % ('Runs/WGAN_FIGR_Results'), score_tr)


	# starting training
	train_hist = {}
	train_hist['D_losses'] = []
	train_hist['G_losses'] = []
	train_hist['per_epoch_ptimes'] = []
	train_hist['total_ptime'] = []
	print('Training start!')
	start_time = time.time()
	for epoch in range(opt.niter):
		print('this is epoch {}'.format(epoch))
		D_losses = []
		G_losses = []

		# learning rate decay
		if (epoch+1) == 11:
			G_optim.param_groups[0]['lr'] /= 10
			D_optim.param_groups[0]['lr'] /= 10
			print("learning rate change!")

		if (epoch+1) == 16:
			G_optim.param_groups[0]['lr'] /= 10
			D_optim.param_groups[0]['lr'] /= 10
			print("learning rate change!")


		### WGAN-GP
		### training discriminator first: [discriminator_pred, discriminator_targets]
		### training generator then; [output, generator_targets]
		print('starting triaing')
		num_iter = 0
		epoch_start_time = time.time()
		print(len(train_loader))
		# for x_, _ in train_loader:
		for i, data in enumerate(train_loader, 0):
			batch_size = data[0].size()[0]
			discriminator_targets = torch.tensor([1] * batch_size + [-1] * batch_size, dtype=torch.float, device=device).view(-1, 1)
			generator_targets = torch.tensor([1] * batch_size, dtype=torch.float, device=device).view(-1, 1)
			if np.shape(data[0])[0]!=batch_size:
				continue

			## training discriminator ##
			G.train()
			#x_ = torch.from_numpy(x_)
			x_=data[0].type(torch.FloatTensor)
			x_ = normalize_data(x_)
			real_batch = Variable(x_.to(device))
			# [32, 100]
			# [32, 3, 64, 64]
			# [32, 3, 64, 64]
			fake_batch = G(torch.tensor(np.random.normal(size=(batch_size, opt.nz)), dtype=torch.float, device=device))
			training_batch = torch.cat([real_batch, fake_batch])

			

			# Training discriminator
			gradient_penalty = calc_gradient_penalty(D, real_batch, fake_batch)
			discriminator_pred = D(training_batch)
			# [64, 3, 64, 64]
			# [64, 1]
			discriminator_loss = wassertein_loss(discriminator_pred, discriminator_targets)
			discriminator_loss += gradient_penalty
			D_losses.append(discriminator_loss)

			D_optim.zero_grad()
			discriminator_loss.backward()
			D_optim.step()

			## Training generator ##
			output = D(G(torch.tensor(np.random.normal(size=(batch_size, opt.nz)), dtype=torch.float, device=device)))
			generator_loss = wassertein_loss(output, generator_targets)
			G_losses.append(generator_loss)

			G_optim.zero_grad()
			generator_loss.backward()
			G_optim.step()


			if i % 10 == 0:
				print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
					  % (epoch, opt.niter, i, len(train_loader),
						 discriminator_loss.item(), generator_loss.item()))
			if i % 100 == 0:
				vutils.save_image(x_,
						'%s/real_samples.png' % outf,
						normalize=True)
				fake = G(fixed_noise)
				vutils.save_image(fake.detach(),
						'%s/fake_samples_epoch_%03d.png' % (outf, epoch),
						normalize=True)
		
		epoch_end_time = time.time()
		per_epoch_ptime = epoch_end_time - epoch_start_time


		print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), opt.niter, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),                                                          torch.mean(torch.FloatTensor(G_losses))))
		p = outf + str(epoch + 1) + '.png'
		if not os.path.isdir(outf):
			os.mkdir(outf)
		writer = SummaryWriter(outf)
		WGAN_show_result((epoch+1), G, p, opt.nz, batch_size, device, writer)
		train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
		train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
		train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

		## evaluation
		score_tr = np.zeros((opt.niter, 4*7+3))
		s = metric.compute_score_raw(opt.dataset, opt.imageSize, dataroot, opt.sampleSize, opt.batchSize, outf+'/real/', outf+'/fake/',\
														 G, opt.nz, conv_model='inception_v3', workers=int(opt.workers))
		score_tr[epoch] = s

	end_time = time.time()
	total_ptime = end_time - start_time
	train_hist['total_ptime'].append(total_ptime)

	print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), opt.niter, total_ptime))
	print("Training finish!... save training results")
	torch.save(G.state_dict(), outf +"/generator_param.pkl")
	torch.save(D.state_dict(), outf +"/discriminator_param.pkl")
	with open(outf+'/train_hist.pkl', 'wb') as f:
		pickle.dump(train_hist, f)

	show_train_hist(train_hist, save=True, path=outf+'/train_hist.png')

	'''
	images = []
	for e in range(train_epoch):
		img_name = 'Runs/WGAN_FIGR_Results/' + str(e + 1) + '.png'
		images.append(imageio.imread(img_name))
	imageio.mimsave('Runs/WGAN_FIGR_Results/generation_animation.gif', images, fps=5)
	'''




