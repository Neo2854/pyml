# CycleGAN
The original cycle GAN paper can be found <a href="https://arxiv.org/pdf/1703.10593.pdf">here</a>.

The <b>cycleGAN.ipynb</b> file consists of cycle GAN implementation on Bitmoji Faces and celebA datasets as the two domains. The Bitmoji Faces can be downloaded from <a href="https://drive.google.com/drive/folders/1_2VtX5D7Nmmu_G5VXRxv4kDqRcvH4Z1z?usp=sharing"></a>.
The celebA dataset used is a simplefied one which in available on <a href = "https://www.kaggle.com/datasets/jessicali9530/celeba-dataset">kaggle</a>.

Downloading the celebA dataset directly from pyTorch is also an option.
```bash
  celebDataset = torchvision.datasets.celebA(root="/path/to/download/dataset/", download=True)
  celebDataloader = DataLoader(celebDataset, batch_size = BATCH_SIZE, shuffle=True, num_workers = 8)
```

The cycleGAN is extramely GPU hungry, so try to decrease batch size or resize images. Use this <a href="https://github.com/soumith/ganhacks">github repo</a> to find tricks about traning GANs.
