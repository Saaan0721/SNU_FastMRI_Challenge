# SNU_FastMRI_Challenge

팀명: Vo

팀원: 김산, 신창민

## Desctription

Mnet을 이용하여 aliased image를 aliasing free image로 변환하는 프로젝트 입니다.

### 1. Image Augmentation

original image, horizontal filped image 두 종류를 사용하여 train data의 수를 2배로 만듭니다.

data를 load하는 과정에서 이루어지며 별도의 directory를 생성하지는 않습니다.

### 2. Data Split

train data와 validatation data를 split할 때 seperate_data.py를 사용하지 않고 load_data.py 내부에서 random_split() 함수를 통해 정해진 비율로 split합니다. (default ratio는 train:validation = 8:2)

``` Python
train_dataset, val_dataset = random_split(data_storage, [train_size, val_size])
```

### 3. Mnet

기본 제공된 Unet을 변형한 5층 Mnet을 사용하였습니다.

## Prerequsite

image augmentation 과정에서 외부 라이브러리(albumentations)를 사용하였습니다.

<https://github.com/albumentations-team/albumentations>

때문에 설치가 필요합니다.

```
pip install -U git+https://github.com/albu/albumentations > /dev/null && echo "All libraries are successfully installed!"
```

## Usage

```
python train.py
python evaluate.py
python (SSIM 측정).py
```

evaluate.py가 생성하는 최종 영상의 directory는 아래와 같습니다.

```
'../result/Mnet/reconstructions_forward/'
```

따라서 SSIM 측정시 directory를 수정하여야 합니다.

예) leaderboard score 측정시 아래와 같이 directory를 지정하였습니다.

```
python leaderboard_eval.py -yp '../result/Mnet/reconstructions_forward/'
```
