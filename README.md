# BA_Semi_Superivsed_Learning


# Semi-Supervised_Learning
 
 ## Semi-Supervised Learning 이란?
 
 적은 labeled data가 있으면서 추가로 활용할 수 있는 대용량의 unlabeled data가 있다면 semi-supervised learning을 고려할 수 있다. Semi-supervised learning (준지도학습)은 소량의 labeled data에는 supervised learning을 적용하고 대용량 unlabeled data에는 unsupervised learning을 적용해 추가적인 성능향상을 목표로 하는 방법론이다. 이런 방법론에 내재되는 믿음은 label을 맞추는 모델에서 벗어나 데이터 자체의 본질적인 특성이 모델링 된다면 소량의 labeled data를 통한 약간의 가이드로 일반화 성능을 끌어올릴 수 있다는 것이다. (아래의 그림에서도 확인할 수 있듯이 왼쪽에 supervised learning과 오른쪽 semi-supervised learning을 비교해보면 supervised learning의 decision boundary는 사실상 optimal하지 않다. unlabeled data를 활용하여 데이터 자체의 분포를 잘 모델링하면 오른쪽 semi-supervised learning 그림처럼 더 좋은 decision boundary를 얻을 수 있다.)
 
 ![image](https://user-images.githubusercontent.com/71392868/209581111-5b34e5e0-5415-494b-8b03-28e269c20005.png)


Semi-supervised learning의 목적함수는 supervised loss L_s 와 unsupervised loss L_u의 합을 최소화하는 것으로 표현할 수 있다. 그 말인즉슨 supervised, unsupervised를 1-stage로 한번에 학습한다. 이것이 2-stage로 이루어지는 self-supervised learning, transfer learning 등과의 차이점이다.


![image](https://user-images.githubusercontent.com/71392868/209581348-98e805b9-93f2-40dc-9646-cf5558b3f763.png)


소량의 labeled data에 적용하는 supervised loss의 경우 target이 discrete value인지 continuous value인지에 따라 classification loss/regression loss를 선택하여 학습하면 된다. 그리고 대용량 unlabeled data는 unsupervised loss를 주어 데이터의 특성에 대해 학습하게 된다. 여기서 unlabeled data에 주는 unsupervised task를 어떻게 정할 것이냐에 따라 방법론이 나뉘게 된다. 다음은 semi-supervised learning의 일반적인 방법론에 대해 살펴보도록하겠다.


## Semi-supervised learning 방법론

https://ainote.tistory.com/6

1. Pseudo Labeling 

![image](https://user-images.githubusercontent.com/71392868/209639831-5db725b9-e20d-4bf0-9361-62ebbaa8ccfe.png)

Pseudo Labeling의 순서는 다음과 같다.

1) Labeled Data로 모델을 학습시킨다.

2) 예열된 모델 (Labeled Data로 학습된 모델) 을 가지고 Unlabeled Data를 예측한다.

3) 예측값들 중 가장 확시한 것에 Pseudo Label을 부여한다.

4) Labeled Data에 Pseudo Label을 가진 데이터를 포함시킨다.

5) 반복

 
이러한 반복 과정을 통해 Labeled Data는 조금씩 늘어가고, Unlabeled Data는 조금씩 줄어간다.

이렇게 두 데이터를 동시에 이용하여 효과적으로 모델을 학습시키는 방법이다..


2. Consistency Regularization

![image](https://user-images.githubusercontent.com/71392868/209640155-a5062174-fd43-4672-bd14-18d536ff370d.png)

Consistency Regularization은 쉽게 말해 "데이터에 가해진 작은 변형은 라벨을 변형시키지 않는다"이다. 왼쪽의 차에 오른쪽과 같이 변형을 가해도 "차"라는 라벨은 변하지 않는다. 그렇다면 모델 또한 동일하게 예측해야 되기 때문에 왼쪽 사진의 예측 확률 분포와 오른쪽 사진의 예측 확률 분포를 동일하게 만들도록 학습한다. 두 확률 분포의 차이가 손실이 되는 것이다. 수식은 아래와 같다.

![image](https://user-images.githubusercontent.com/71392868/209640179-9888b632-805a-473e-82f3-7042dbd124eb.png)


3. Entropy Minimization

![image](https://user-images.githubusercontent.com/71392868/209640197-6d05d920-2274-4dc8-a591-bbf2d24b6acf.png)

Entropy Minimization은 "좀 더 확실하게 만들기"라고 이해하면 될 것  같다. 한 이미지에 대한 예측 확률 분포가 Dog : Cat = 0.7 : 0.3이었다면, 위 과정을 거치면 Dog : Cat = 0.9 : 0.1과 같이 변하며, 이 값을 학습에 사용한다. 흔히 아는 Entropy와 좀 비슷한 감이 있다. 이러한 과정을 거치면 모델 학습 속도가 빨라지며, 더 정확하게 학습되는 효과를 가진다. 하지만 반대로, 이상한 라벨로 엔트로피가 최소화된다면, 문제가 발생한다.


## Semi-supervised learning Model

### 1. Ladder Network

다음은 Ladder Network의 구조를 나타냅니다. corrupted path, clean path, denoising path로 구성되어 있으며 학습 또한 지도학습과 비지도학습이 결합되어 진행됩니다.

![image](https://user-images.githubusercontent.com/71392868/209638593-9b5227e8-60b6-41c9-99de-f8ed7b4d787e.png)

Ladder Network를 모델에 도입하는 과정은 세 단계로 이루어진다.

1: Encoder로 지도 학습을 하는 feedforward 모델 구축

feedforward 모델에서는 지도학습과 비지도학습이 병렬적으로 진행된다. 
논문에서는 Batch normalization을 통해 BN이 주는 효과인 covariance shift를 줄이는 동시에 일반화 가정을 만족시킬 수 있다고 언급하므로 BN을 필수적으로 사용한다. Feedforward 모델 구축을 통해서 지도학습과 비지도학습이 병렬적으로 진행되며 지도학습 측면에서는 corrupted path에서 생성된 output이 실제 target과 유사하도록 학습이 진행된다. 아래그림의 파란색으로 표시된 지도학습에 해당하는 부분이다.

2: 각 층과 mapping하고 비지도학습을 돕는 decoder 구축

두 번째 단계에서는 비지도학습이 진행된다. 즉 1번 단계에서 encoder단에서 학습된 비지도학습 가중치가 decoder단으로 내려오면서 아래층의 잠재변수 학습에 영향을 주고 동시에 수평적으로 연결된 corrupted path의 같은 층의 정보가 영향을 주면서 clean path의 z와 유사하도로 학습이 진행된다..

3: 모든 손실합수의 합을 최소화하는 Ladder Network 학습

지도학습의 loss function과 비지도학습의 loss function을 합친 최종 loss function이 작아지도록 학습이 진행된다.



### 2. Temporal Ensembel


Π-model에서 얻어진 target값은 nosiy하기 때문에 temporal ensemble은 이전 network들의 결과를 사용함으로서 이러한 문제를 완화하고자 한다. Temporal ensembling의 structure는 아래와 같다


![image](https://user-images.githubusercontent.com/71392868/209638223-ddb5655f-5912-4be3-8086-2bc1ba98486c.png)

 매 학습 에폭마다 network의 output z는 ensemble output Z로 축적된다. Z는 Zi ← αZi + (1 − α)zi으로 업데이트 되며 α는 training history 관점에서 얼마나 떨어졌는지를 조절하는 momentum term 이다. 이전에 적용하는 dropout과 augmentation으로 인해 나온 결과와 ensemble 결과들을 weighted average한 상황이다. 또한, 최근 에폭의 결과에는 더 큰 weight가 포함되어 있을 것이다. 

 Π-model과 다른 점은 Π-model에서는 해당 시점에서 unsupervised target를 만들지만 temporal ensembling에서는 이전 ensemble의 결과들을 unsupervised의 target으로 사용한다는 점이다. 첫 에폭에는 ensemble output Z가 없기 때문에 w(t)과 함께 0으로 설정한다.  이러한 ensemble의 효과를 통해 noisy한 부분들을 제거하고 성능을 올릴 수 있었다.




### 3. FixMatch


아래그림은 FixMatch의 전반적인 프레임워크 이다. 단순하게, 한 이미지를 서로 다른 방식으로 변형하여 대조하는 방식으로 학습이 진행된다. FixMatch에서 핵심적인 부분은 Pseudo Labeling과 Consistency Regularization을 결합했다는 점이다.

![image](https://user-images.githubusercontent.com/71392868/209638975-9ab7ecab-a0b3-49d7-802b-3e68ab1a205f.png)


# Experiment Dataset

1. Mnist Dataset


MNIST 데이터셋은 28x28 사이즈의 손글씨 데이터셋이다. 0~9까지 총 10개의 클래스를 가지고 있으며, 색상 채널이 없는 흑백 이미지이다. 비교적 단순한 예제이기 때문에 머신러닝, 딥러닝 기초 예제로 많이 사용된다.

![image](https://user-images.githubusercontent.com/71392868/209644183-aa5b5440-a228-4f3f-bdc0-682e1126e77f.png)






2. Cifar-10 Dataset

CIFAR-10 dataset은 32x32픽셀의 60000개 컬러이미지가 포함되어있으며, 각 이미지는 10개의 클래스로 라벨링이 되어있다.
또한, MNIST와 같이 머신러닝 연구에 가장 널리 사용되는 dataset중 하나이다.

![image](https://user-images.githubusercontent.com/71392868/209644242-13729f7c-4ec1-4d8a-ac69-990aa1c2d4fd.png)


60000개 중, 50000개 이미지는 트레이닝 10000개 이미지는 테스트용도로 사용된다.




Ref 1 : https://gruuuuu.github.io/machine-learning/cifar10-cnn/

Ref 2 : https://github.com/9310gaurav/virtual-adversarial-training/blob/master/main.py

Ref 3 : https://sanghyu.tistory.com/177

Ref 4 : https://woosikyang.github.io/Ladder-Network.html

Ref 5 : https://m.blog.naver.com/winddori2002/222162725269


