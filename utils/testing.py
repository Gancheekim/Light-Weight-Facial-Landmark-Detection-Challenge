import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def cal_loss(landmarks, label, H=384, W=384):
	# loss = landmarks136 - groundtruth136
	# loss = np.sqrt(np.sum(np.power(loss,2), 1))
	# loss = np.mean(loss)
	# return loss/((H*W)**0.5)
	# x1 = landmarks[:, 0::2].clone()
	# y1 = landmarks[:, 1::2].clone()

	# x2 = label[:, 0::2].clone()
	# y2 = label[:, 1::2].clone()
	x1 = np.copy(landmarks[:, 0::2])
	y1 = np.copy(landmarks[:, 1::2])

	x2 = np.copy(label[:, 0::2])
	y2 = np.copy(label[:, 1::2])

	x_diff = np.square(x2 - x1)
	y_diff = np.square(y2 - y1)
	d = (H*W)**0.5

	nme = x_diff + y_diff
	nme = np.power(nme, 0.5)
	nme /= d
	nme = np.mean(nme,1)
	nme = np.sum(nme)

	return nme


def cal_loss1(landmarks, label, H=384, W=384):
	# loss = landmarks136 - groundtruth136
	# loss = np.sqrt(np.sum(np.power(loss,2), 1))
	# loss = np.mean(loss)
	# return loss/((H*W)**0.5)
	# x1 = landmarks[:, 0::2].clone()
	# y1 = landmarks[:, 1::2].clone()

	# x2 = label[:, 0::2].clone()
	# y2 = label[:, 1::2].clone()

	x1 = np.copy(landmarks[:, 0::2])
	y1 = np.copy(landmarks[:, 1::2])

	x2 = np.copy(label[:, 0::2])
	y2 = np.copy(label[:, 1::2])

	# x_diff = torch.square(x2 - x1)
	# y_diff = torch.square(y2 - y1)

	x_diff = (x2 - x1)**2
	y_diff = (y2 - y1)**2
	# d = (H*W)**0.5
	d = 384

	# nme = x_diff + y_diff
	# nme = torch.pow(nme, 0.5)
	# nme /= d
	# nme = torch.mean(nme,1)
	# nme = torch.sum(nme)

	nme = ((x_diff + y_diff)**0.5)/d
	nme = np.sum(np.mean(nme,1))
	return nme


landmarks = np.random.randint(384, size=(64,136*10))
label = np.random.randint(384, size=(64,136*10))

a1 = time.time()
nme = cal_loss(landmarks, label)
total1 = time.time() - a1
print(f'elapsed: {total1}')

a2 = time.time()
nme = cal_loss1(landmarks, label)
total2 = time.time() - a2
print(f'elapsed: {total2}')

# a = cv2.imread('D:\\Users\\Lenovo\\Downloads\\a.jpg')
a = Image.open('D:\\Users\\Lenovo\\Downloads\\a.jpg').convert('RGB')
w, h = a.size
print(a.size)


# b = np.zeros((w,h,3))
# for m in range(w):
# 	for n in range(h):
# 		x = m + 2*((w/2)-1 - m)
# 		# print(x)
# 		b[int(x),n,:] = a[m,n,:]

b = a.rotate(30)

cx = [10, 200, 300, 400]
cy = [10, 200, 300, 400]

dx = [40, 1023, 666]
dy = [80, 777, 1000]

plt.figure()
plt.imshow(np.asarray(a)/255)
plt.imshow(np.asarray(b)/255)
for i in range(4):
	plt.scatter(cx[i], cy[i], c='#1f77b4', s=5)
for i in range(3):
	plt.scatter(dx[i], dy[i], c='#d62728', s=5)
# plt.clf()
plt.show()