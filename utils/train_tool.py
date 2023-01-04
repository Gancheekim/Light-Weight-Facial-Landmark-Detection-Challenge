import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import time
import os
import matplotlib.pyplot as plt
# from PIL import Image
# from torchvision.transforms import transforms


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


def create_visualize(test_img, predict1, groundtruth1):
	print('saving some results of validation set...')
	plt.figure()
	for i in range(min(len(test_img),10)):
		test = test_img[i, :,:,:]
		predict = predict1[i, :]
		groundtruth = groundtruth1[i, :]
		x = predict[0::2]
		y = predict[1::2]
		x1 = groundtruth[0::2]
		y1 = groundtruth[1::2]
		# plt.figure()
		plt.imshow(test)
		for index in range(len(x)):
			plt.scatter(x[index], y[index], c = '#1f77b4', s = 5) 
			plt.scatter(x1[index], y1[index], c = '#d62728', s = 5) 
		plt.savefig('test_image/test'+str(i)+'.png')
		plt.clf()
		

def convert_to_tensor(label):
	new_label = torch.zeros((len(label[0]), 136))
	for i in range(136):
		new_label[:, i] = label[i].t()
	return new_label

def cal_loss(landmarks, label, H=384, W=384): # NME
	# loss = landmarks136 - groundtruth136
	# loss = np.sqrt(np.sum(np.power(loss,2), 1))
	# loss = np.mean(loss)
	# return loss/((H*W)**0.5)
	x1 = landmarks[:, 0::2].clone()
	y1 = landmarks[:, 1::2].clone()

	x2 = label[:, 0::2].clone()
	y2 = label[:, 1::2].clone()

	x_diff = torch.square(x2 - x1)
	y_diff = torch.square(y2 - y1)
	# d = (H*W)**0.5
	d = 384

	nme = x_diff + y_diff
	nme = torch.pow(nme, 0.5)
	nme /= d
	nme = torch.mean(nme,1)
	nme = torch.sum(nme)
	# print(nme.size())
	return nme

# def gnll(landmarks, label, loss, device): # NME
# 	##size:(38, 68)
# 	x1 = landmarks[:, 0::2].clone()
# 	y1 = landmarks[:, 1::2].clone()

# 	x2 = label[:, 0::2].clone()
# 	y2 = label[:, 1::2].clone()

# 	input = torch.cat((x1, y1), 1)
# 	input = input.to(device)
# 	target = torch.cat((x2, y2), 1)
# 	target = target.to(device)
# 	var = torch.ones(38, 136, requires_grad=True)
# 	var = var.to(device)	

# 	g_loss = loss(input, target, var)

# 	return g_loss

def train(model, train_loader, val_loader, num_epoch, save_path, device, scheduler, optimizer):
	best_loss = np.inf
	# criterion = AdaptiveWingLoss()
	# criterion = nn.MSELoss()
	l_func = nn.GaussianNLLLoss()

	for epoch in range(num_epoch):
		print(f'epoch = {epoch}')
		start_time = time.time()
		train_loss = 0.0

		# start train		
		model.train()
		for batch_idx, ( data, label,) in enumerate(tqdm(train_loader)):
			# if batch_idx > 5:
				# break

			# put the data and label on the device
			# note size of data (B,C,H,W) --> B is the batch size
			data = data.to(device)
			# print(data)
			label = convert_to_tensor(label)
			label = label.to(device)

			# test = gnll(label, label)
			# pass forward function define in the model and get output
			landmarks = model(data)
			# landmarks.requires_grad_(True)
			
			# calculate loss
			loss = cal_loss(landmarks, label)
			# loss = criterion(landmarks, label)
			loss.requires_grad_(True)

			# discard the gradient left from former iteration
			optimizer.zero_grad()

			# calculate the gradient from the loss function
			loss.backward()

			# if the gradient is too large, we dont adopt it
			# grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)

			# Update the parameters according to the gradient we calculated
			optimizer.step()

			train_loss += loss

		scheduler.step()

		# averaging training_loss and calculate accuracy
		train_loss = train_loss.item() / len(train_loader.dataset)
		
		# start validating
		with torch.no_grad():
			model.eval()
			val_loss = 0

			for batch_idx, ( val_data, val_label,) in enumerate(tqdm(val_loader)):
				val_data = val_data.to(device)
				val_label = convert_to_tensor(val_label)
				val_label = val_label.to(device)

				val_output = model(val_data)
				loss = cal_loss(val_output, val_label)
				# loss = criterion(val_output, val_label)
				val_loss += loss

		val_loss = val_loss.item() / len(val_loader.dataset) 

		# display result
		end_time = time.time()
		elp_time = end_time - start_time
		min = elp_time // 60 
		sec = elp_time % 60
		print('*'*10)
		print('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC '.format(elp_time // 60, elp_time % 60, (end_time-start_time) // 60, (end_time-start_time) % 60))
		print(f'training loss : {train_loss:.4f} ')
		print(f'val loss : {val_loss:.4f} ')


		# save the best model if it gain performance on validation set
		if  val_loss < best_loss:
			print('==== best loss, update parameters! ====\n')
			best_loss = val_loss
			torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
			torch.save(optimizer.state_dict(), os.path.join(save_path, 'best_model_optimizer.pt'))
			torch.save(scheduler.state_dict(), os.path.join(save_path, 'best_model_scheduler.pt'))
			# create visualization of predicted landmarks
			if epoch == num_epoch-1 or (epoch % 2 == 0 and epoch != 0):
				test_img = val_data.detach().cpu().numpy()
				predict = val_output.detach().cpu().numpy()
				groundtruth = val_label.detach().cpu().numpy()
				test_img = test_img.swapaxes(1,2)
				test_img = test_img.swapaxes(2,3)
				create_visualize(test_img, predict, groundtruth)
				print('check out test image to see landmarks!\n')
		else:
			print('=====================================\n')
