import torch
import os
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm

def load_parameters(model, path):
	print(f'Loading model parameters from {path}...')
	param = torch.load(path, map_location='cuda:0')
	model.load_state_dict(param)
	print("End of loading !!!")


def test_result(testset_path, model, device):
	means = [0.485, 0.456, 0.406]
	stds = [0.229, 0.224, 0.225]

	f = open("solution.txt", "w")
	images = []
	filenames = []
	# read all the test data 
	for filename in os.listdir(testset_path):
		img_path = os.path.join(testset_path, filename)
		filenames.append(filename)
		img = Image.open(img_path).convert('RGB')
		test_transform = transforms.Compose([
							transforms.ToTensor(),
							transforms.Normalize(means, stds),
						])
		img = test_transform(img)
		img = torch.reshape(img, (1, 3, 384, 384))
		images.append(img)

	model.eval()
	print('start predicting test set...')
	with torch.no_grad():
		for idx, img in enumerate(tqdm(images)):
			img = img.to(device)
			test_output = model(img)
			coords = test_output.to('cpu').numpy()
			coords = test_output[0]
			line = str(filenames[idx]) + " "
			for i in range(136):
				line += str(round(coords[i].item(),4))
				if i < 135:
					line += " "
				else:
					line += "\n"
			f.write(line)
	f.close()