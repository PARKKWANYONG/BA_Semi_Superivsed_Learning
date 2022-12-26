# BA_Semi_Superivsed_Learning

ref 1 :   https://sanghyu.tistory.com/177

# Semi-Supervised_Learning
 
 ## Semi-Supervised Learning 이란?
 
 적은 labeled data가 있으면서 추가로 활용할 수 있는 대용량의 unlabeled data가 있다면 semi-supervised learning을 고려할 수 있다. Semi-supervised learning (준지도학습)은 소량의 labeled data에는 supervised learning을 적용하고 대용량 unlabeled data에는 unsupervised learning을 적용해 추가적인 성능향상을 목표로 하는 방법론이다. 이런 방법론에 내재되는 믿음은 label을 맞추는 모델에서 벗어나 데이터 자체의 본질적인 특성이 모델링 된다면 소량의 labeled data를 통한 약간의 가이드로 일반화 성능을 끌어올릴 수 있다는 것이다. (아래의 그림에서도 확인할 수 있듯이 왼쪽에 supervised learning과 오른쪽 semi-supervised learning을 비교해보면 supervised learning의 decision boundary는 사실상 optimal하지 않다. unlabeled data를 활용하여 데이터 자체의 분포를 잘 모델링하면 오른쪽 semi-supervised learning 그림처럼 더 좋은 decision boundary를 얻을 수 있다.)
 
 ![image](https://user-images.githubusercontent.com/71392868/209581111-5b34e5e0-5415-494b-8b03-28e269c20005.png)


Semi-supervised learning의 목적함수는 supervised loss L_s 와 unsupervised loss L_u의 합을 최소화하는 것으로 표현할 수 있다. 그 말인즉슨 supervised, unsupervised를 1-stage로 한큐에 학습한다. 이것이 2-stage로 이루어지는 self-supervised learning, transfer learning 등과의 차이점이다.


![image](https://user-images.githubusercontent.com/71392868/209581348-98e805b9-93f2-40dc-9646-cf5558b3f763.png)


소량의 labeled data에 적용하는 supervised loss의 경우 target이 discrete value인지 continuous value인지에 따라 classification loss/regression loss를 선택하여 학습하면 된다. 그리고 대용량 unlabeled data는 unsupervised loss를 주어 데이터의 특성에 대해 학습하게 된다. 여기서 unlabeled data에 주는 unsupervised task를 어떻게 정할 것이냐에 따라 방법론이 나뉘게 된다. 이는 semi-supervised learning의 일반적인 가정을 먼저 살펴본 후에 자세히 설명하도록 하겠다.


## Semi-supervised learning 방법론들

1. Entropy minimization 

2. Proxy-label method

3. Generative models

4. Consistency regularization (Consistency training)

5. Holistic methods

6. Graph-based methods 
등등 다양하지만 Consistency Regularization만 다루도로가겠음.


Consistency regularization (Consistency training) [6-9]

이 방법은 unlabeled data point에 작은 perturbation을 주어도 예측의 결과에는 일관성이 있을 것이다라는 가정에서 출발한다. unlabeled data는 예측결과를 알 수 없기 때문에 data augmentation을 통해 class가 바뀌지 않을 정도의 변화를 줬을 때, 원 데이터와의 예측결과가 같아지도록 unsupervised loss를 주어 학습하게 된다. 이를 통해 약간 헷갈리는 샘플들에 대해 class를 유연하게 예측할 수 있도록 해준다. 성능이 좋은 semi-supervised learning 모델들은 대체로 consistency regularization을 사용하고 있다. 가장 유명한 모델으로는 Ladder Network, Temporal Ensemble, Virtual Adversarial Training (VAT) 등이 있다. 이미지 분야의 경우 많은 연구들을 통해 class가 바뀌지 않을 정도의 data augmentation 기법들이 많이 연구되었지만 그 외의 도메인에서는 도메인 지식없이는 제대로된 data augmentation을 적용하기 어려워 적용에 한계가 있다.

1. Ladder Network




2. Temporal Ensembel



3. Virtual Adversarial Training (VAT)
