# Computer Vision EEE 5053 (Spring 2022)
<font size="5">Final Project: Light-Weight Facial Landmark Detection Challenge</font>

### members:
- 顏子鈞 Gan Chee Kim
- 王譽凱 Ong Ee Khai
- 沈驁毅 Sim Ao Yi

## Outline
1. [Problem Description](#Problem-Description)
2. [Overview of Method](#Overview-of-Method)
3. [Visualization](#Visualization)
4. [Comparisons of other method attempts](#Comparisons-of-other-method-attempts)
5. [References](#References)


## 1. Problem Description

## 2. Overview of Method
### 1.1 Backbone network
> 為了滿足此次的挑戰：輕量且精準的facial landmark detection，我們使用的backbone架構是由Google提出的MobileNetV3<sup>[1]</sup>，實作程式碼<sup>[3]</sup>則是從Github上的open resource上修改而來。為了達到讓model大小<15mb的要求，我們將原作中的某一層layer給去除，最終我們的model parameter總共為13.83mb。此外，由於本次的任務是predict 68個facial landmark，因此我們將model最後的output修改成一個大小為1x136的torch，分別對應landmark的68組x和y坐標。  
<p align="center">
    <img src="./readme_img/fig1.jpg" alt="drawing" width="400"/><br> </br>
</p>  

### 1.2 Training Details
> 我們在training時使用的最佳化optimizer是*AdamW*，詳細的spec為:  

|betas        |eps  | weight_decay|
|-------------|-----|-------------|
|(0.9, 0.999) |1e-8 |0.0001       |  
 

> 此外，初始的learning rate為1e-3，但我們使用multistep的方式來讓learning rate隨著不同的epoch而下降，進而達到讓初始訓練速度加快、後續的訓練又能更精細的表現。詳細的spec為:  

|Epochs to decrease learning rate |Decrease rate (new LR = old LR * rate) |
|----------|-----|
|5, 10, 15 |0.45 |    

> 出於會讓照片的資訊量下降的考量，因此我們沒有resize圖片，也沒有打算用grayscale來訓練，因此input size是(384x384x3)，對於一個network來說這樣的input size並不小。而經過反復試驗後，我們得出rtx 2080也大概只能在一個batch塞下38張左右的照片，因此我們將batch size設為38，最終訓練30個epoch。  
至於Loss function，我們一樣是採用Normalized mean error (NME)，我們會一次過計算一整個batch的loss，然後加起來，再backpropagate回去更新network的參數。

### 1.3 Data Processing & Augmentation  

| Data Augmentation | Parameter Used |
|-------------------|----------------|
|ColorJitter        | Brightness = 0.1, Contrast = 0.1, Saturation = 0.1 |
|GaussianBlur       | Kernel_size = (3,3), Sigma = (0.1, 2) |
|RandomGrayScale    | P = 0.2 |
|RandomErasing      |P = 0.2, Scale = (0.08, 0.25), Ratio = (0.8, 3.3)|
|Normalization      | Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225] |

> 我們對於training data一共使用了四種data augmentation<sup>[2]</sup>, 其中GaussianBlur以及RandomErasing對於model訓練的效果有顯著的提升。此外，我們也參考了網路上非常著名的ImageNet的normalization method，而normalize資料後再訓練確實對network的prediction有幫助。

## 3. Visualization
### 3.1 Training Data
> 以下是針對training data在經過不同的變化之後，將其visualize以及與結果的對比圖：  

<img src="./readme_img/1.jpg" alt="drawing" width="400"/><br> </br>  

> 在以上的例子中採用的是RandomGrayScale和GaussianBlur，將原始圖片轉為灰階，並且通過減少像素之間的差異值達到模糊的效果，用意在於增加對於灰階或是模糊圖片的判斷能力。 

<img src="./readme_img/2.jpg" alt="drawing" width="400"/><br> </br>

> 這個例子中則是使用了ColorJitter和RandomErasing。ColorJitter將原始圖片的亮度、對比度和飽和度根據參數進行調整。而這邊使用的RandomErasing機率性地將圖片的部分區域按照比例去除，嘗試增加model對於部分面部被遮蔽的圖片的預測能力。

### 3.2 Validation Data
<img src="./readme_img/3.jpg" alt="drawing" width="400"/><br> </br>

### 3.3 Testing Data
### 3.3.1 Normal and Clear Faces
### 3.3.2 Edge-cases Faces: Occlusion-involved
### 3.3.3 Edge-cases Faces: Mask/Make-up
### 3.3.4 Edge-cases Faces: Weired-angle
### 3.3.5 Edge-cases Faces: Bad illumination 
### 3.3.6 Edge-cases Faces: Multiple Faces


## 4. Comparisons of other method attempts
### 4.1 Data Augmentation

### 4.2 Other optimizers and learning rate scheduling


## 5. Reference