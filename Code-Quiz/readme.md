# PART 1: 기초 다지기 (1~9차시)

---

## 1차시: 과정 소개 및 AI 시대의 빅데이터

### 실습 코드 예제  
(내용 주로 개념 이므로, 빅데이터 사례 뉴스 기사 API 호출 예시)

```python
import requests

url = "https://newsapi.org/v2/top-headlines"
params = {
    'apiKey': 'YOUR_NEWSAPI_KEY',
    'category': 'technology',
    'language': 'en',
    'pageSize': 5
}
response = requests.get(url, params=params)
news_items = response.json()['articles']

for i, item in enumerate(news_items):
    print(f"{i+1}. {item['title']}\n   Source: {item['source']['name']}\n")
```

### 참고문헌 및 추가 자료  
- «Big Data: Principles and best practices», Rajkumar Buyya  
- 빅데이터 5V와 산업 사례 정리: https://www.sas.com/en_us/insights/big-data/what-is-big-data.html  
- AI 발전 과정: https://ai.google/education/

### 퀴즈 문제  
1. 빅데이터 5V 중 ‘Veracity’는 무엇을 의미하는가?  
   A) 데이터 용량  
   B) 데이터 신뢰성  
   C) 데이터 속도  
   D) 데이터 다양성

2. 생성형 AI가 가장 최근에 주목받기 시작한 이유는?  
   A) 컴퓨터 속도 증가  
   B) 대용량 데이터와 모델 구조 혁신  
   C) 고효율 전력소비  
   D) 전통 통계학 이론 발전

---

## 2차시: 데이터 분석을 위한 개발 환경 구축

### 실습 코드 예제

```bash
# Anaconda 가상환경 생성 터미널 명령
conda create -n myenv python=3.9
conda activate myenv

# Git 기본 명령
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/username/repo.git
git push -u origin main
```

### 참고문헌 및 추가 자료  
- Anaconda 공식 홈페이지 https://www.anaconda.com/  
- Git & GitHub 학습 자료 https://learngitbranching.js.org/  
- Google Colab 사용법: https://colab.research.google.com/notebooks/intro.ipynb

### 퀴즈 문제  
1. Jupyter Notebook에서 커널을 재시작하는 단축키는?  
   A) Ctrl + Enter  
   B) Shift + Enter  
   C) 0, 0 (0 연속 두 번)  
   D) Ctrl + Shift + R

2. Git 명령어 중 저장소 복사하는 것은?  
   A) commit  
   B) clone  
   C) push  
   D) merge

---

## 3차시: Python 프로그래밍 핵심 기초

### 실습 코드 예제

```python
# 간단한 함수 예제
def greet(name):
    return f"Hello, {name}!"

print(greet("Data Scientist"))

# 클래스와 메서드 예제
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
    
    def deposit(self, amount):
        self.balance += amount
        print(f"{amount} deposited. New balance is {self.balance}.")
    
    def withdraw(self, amount):
        if amount > self.balance:
            print("Insufficient funds")
        else:
            self.balance -= amount
            print(f"{amount} withdrawn. New balance is {self.balance}.")

account = BankAccount("Alice", 100)
account.deposit(50)
account.withdraw(30)
```

### 참고문헌 및 추가 자료  
- «Python Crash Course», Eric Matthes  
- Python 공식 튜토리얼: https://docs.python.org/3/tutorial/  
- 객체지향 프로그래밍 소개: https://realpython.com/python3-object-oriented-programming/

### 퀴즈 문제  
1. 파이썬에서 함수는 어떤 키워드로 정의하는가?  
   A) func  
   B) def  
   C) function  
   D) lambda

2. 클래스의 생성자 메서드는?  
   A) __init__  
   B) __start__  
   C) __main__  
   D) __self__

---

## 4차시: 데이터 분석 라이브러리: NumPy

### 실습 코드 예제

```python
import numpy as np

# 배열 생성과 연산
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("Sum:", a + b)

# 배열 재구조화
c = np.arange(12).reshape(3,4)
print(c)

# 브로드캐스팅 예제
d = np.array([1, 2, 3])
print(c + d)
```

### 참고문헌 및 추가 자료  
- NumPy 공식 매뉴얼: https://numpy.org/doc/  
- «Python for Data Analysis», Wes McKinney (NumPy 챕터)  
- 튜토리얼: https://www.datacamp.com/community/tutorials/python-numpy-tutorial

### 퀴즈 문제  
1. np.arange(10) 함수는 무엇을 반환하는가?  
   A) 0부터 9까지의 배열  
   B) 1부터 10까지의 배열  
   C) 0부터 10까지의 배열  
   D) 랜덤 배열

2. 브로드캐스팅이 일어나려면 배열의 어떤 조건이 필요한가?  
   A) 차원의 수가 같아야 한다  
   B) 크기가 똑같아야 한다  
   C) 한 쪽이 스칼라이거나 차원 수가 같거나 1이어야 한다  
   D) 모든 값이 같아야 한다

---

## 5차시: 데이터 분석 라이브러리: Pandas

### 실습 코드 예제

```python
import pandas as pd

# DataFrame 생성
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 22]}
df = pd.DataFrame(data)

# CSV 파일 읽기/쓰기
df.to_csv('test.csv', index=False)
df2 = pd.read_csv('test.csv')

# 인덱싱과 필터링
print(df.loc[0, 'Name'])
print(df[df['Age'] > 23])
```

### 참고문헌 및 추가 자료  
- Pandas 공식 문서 https://pandas.pydata.org/pandas-docs/stable/  
- «Python for Data Analysis», Wes McKinney (Pandas 핵심 부분)

### 퀴즈 문제  
1. Pandas DataFrame에서 특정 컬럼 선택하는 방법은?  
   A) df[‘컬럼명’]  
   B) df.select(‘컬럼명’)  
   C) df.column(‘컬럼명’)  
   D) df.loc(‘컬럼명’)

2. CSV 파일 읽기는 어떤 함수인가?  
   A) pd.readfile()  
   B) pd.opencsv()  
   C) pd.read_csv()  
   D) pd.load_csv()

---

## 6차시: Pandas를 활용한 데이터 정제 및 변환

### 실습 코드 예제

```python
import pandas as pd
import numpy as np

data = {'Name': ['Tom', 'Nick', 'Krish', None], 'Age': [20, 21, None, 22]}
df = pd.DataFrame(data)

# 결측치 확인 및 처리
print(df.isnull().sum())
df['Name'].fillna('Unknown', inplace=True)
df['Age'].fillna(df['Age'].mean(), inplace=True)

# 이상치 처리
df.loc[df['Age'] > 100, 'Age'] = df['Age'].mean()

# 그룹화 및 집계
grouped = df.groupby('Name').agg({'Age': 'mean'})
print(grouped)
```

### 참고문헌 및 추가 자료  
- Pandas 공식 문서: Missing Data https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html  
- «Data Wrangling with Pandas» https://jakevdp.github.io/PythonDataScienceHandbook/03.04-missing-values.html  

### 퀴즈 문제  
1. 결측치 확인 함수는?  
   A) isnull()  
   B) nullcheck()  
   C) checkna()  
   D) nanvalidate()

2. 그룹화 함수는 무엇인가?  
   A) group()  
   B) groupby()  
   C) aggregate()  
   D) summarize()

---

## 7차시: AI를 위한 핵심 선형대수

### 실습 코드 예제

```python
import numpy as np

# 벡터와 행렬 생성
v = np.array([1, 2, 3])
M = np.array([[1, 2], [3, 4]])

# 행렬 곱셈
print(np.dot(M, v[:2]))

# 고유값, 고유벡터 계산
eigvals, eigvecs = np.linalg.eig(M)
print('Eigenvalues:', eigvals)
print('Eigenvectors:\n', eigvecs)
```

### 참고문헌 및 추가 자료  
- Gilbert Strang, «Introduction to Linear Algebra»  
- Khan Academy Linear Algebra 강의 https://www.khanacademy.org/math/linear-algebra  
- NumPy linalg docs https://numpy.org/doc/stable/reference/routines.linalg.html  

### 퀴즈 문제  
1. 행렬 곱셈이 가능한 두 행렬의 조건은?  
   A) 첫 행렬 열의 수와 둘째 행렬 행의 수가 같다  
   B) 두 행렬 모두 정방행렬이어야 한다  
   C) 두 행렬 차원이 동일해야 한다  
   D) 첫 행렬 행의 수와 둘째 행렬 열의 수가 같다

2. 고유값(Eigenvalue)은 무엇을 의미하는가?  
   A) 행렬의 대각선 요소  
   B) 선형 변환에서 벡터 방향을 유지하는 스케일 값  
   C) 벡터의 크기  
   D) 벡터 합

---

## 8차시: AI를 위한 핵심 확률 및 통계

### 실습 코드 예제

```python
import numpy as np
from scipy import stats

# 정규분포 난수 생성 및 시각화
data = np.random.normal(loc=0, scale=1, size=1000)

# 평균, 분산 계산
print("Mean:", np.mean(data))
print("Variance:", np.var(data))

# t-test 예제
group1 = np.random.normal(5, 1, 30)
group2 = np.random.normal(5.5, 1, 30)
t_stat, p_val = stats.ttest_ind(group1, group2)
print("t-statistic:", t_stat, "p-value:", p_val)
```

### 참고문헌 및 추가 자료  
- «Think Stats», Allen B. Downey  
- SciPy stats 문서 https://docs.scipy.org/doc/scipy/reference/stats.html  
- Khan Academy Statistics https://www.khanacademy.org/math/statistics-probability  

### 퀴즈 문제  
1. 중심극한정리(central limit theorem)는 무엇을 설명하는가?  
   A) 큰 표본의 평균값 분포가 정규분포에 근접함  
   B) 분산이 커진다는 의미  
   C) 표본이 많아지면 결과가 임의가 됨  
   D) 확률의 총합은 항상 1임

2. t-test의 p-value가 0.03이라면 보통 유의미하다고 판단하는 유의 수준은?  
   A) 0.1  
   B) 0.05  
   C) 0.01  
   D) 0.001

---

## 9차시: 데이터 시각화

### 실습 코드 예제

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 데이터 생성
data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'],
    'Values': [3, 7, 5, 9]
})

# Matplotlib 막대그래프
plt.bar(data['Category'], data['Values'])
plt.title('Bar Chart Example')
plt.show()

# Seaborn 히트맵
flights = sns.load_dataset("flights").pivot("month", "year", "passengers")
sns.heatmap(flights, annot=True)
plt.show()
```

### 참고문헌 및 추가 자료  
- Matplotlib 공식 문서: https://matplotlib.org/stable/contents.html  
- Seaborn 튜토리얼: https://seaborn.pydata.org/tutorial.html  
- «Python Data Science Handbook» (Visualization 장)  

### 퀴즈 문제  
1. Seaborn에서 히트맵을 만들 때 기본적으로 사용되는 함수명은?  
   A) sns.barplot()  
   B) sns.heatmap()  
   C) sns.scatterplot()  
   D) sns.lineplot()

2. Matplotlib에서 그래프 제목을 추가하는 함수는?  
   A) plt.title()  
   B) plt.xlabel()  
   C) plt.ylabel()  
   D) plt.legend()

---
네! PART 2 (Weeks 4-7) 각 차시별로 실습 코드 예제, 참고문헌, 추가 자료, 퀴즈 문제 포함 상세 내용을 제작해 드립니다.  
예시로 10차시부터 14차시까지 한꺼번에 보여드리겠습니다.

---

# PART 2: 핵심 기술 습득 (Weeks 4-7)

---

## 10차시: 데이터 수집 기술

### 실습 코드 예제

```python
import requests
from bs4 import BeautifulSoup

# 웹 스크레이핑 예: Hacker News 첫 페이지 뉴스 제목 크롤링
url = "https://news.ycombinator.com/"
res = requests.get(url)
soup = BeautifulSoup(res.text, "html.parser")
titles = soup.select(".titleline > a")

for i, title in enumerate(titles[:5], 1):
    print(f"{i}. {title.text}")

# API 요청 예: OpenWeatherMap 날씨 데이터 (API 키 필요)
api_key = 'YOUR_API_KEY'
city = 'Seoul'
url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"

response = requests.get(url)
print(response.json())
```

---

### 참고문헌 및 추가 자료

- «Web Scraping with Python» by Ryan Mitchell  
- BeautifulSoup 공식 문서: https://www.crummy.com/software/BeautifulSoup/bs4/doc/  
- Requests 라이브러리 공식 문서: https://docs.python-requests.org/en/latest/  
- OpenWeatherMap API 설명: https://openweathermap.org/api  

---

### 퀴즈 문제

1. `requests.get()` 함수가 반환하는 객체의 타입은?  
   A) string  
   B) Response  
   C) dict  
   D) list  

2. HTML 문서 내에서 특정 CSS 클래스명을 가진 엘리먼트를 찾으려면 BeautifulSoup 메서드 중 무엇을 사용하는가?  
   A) `select()`  
   B) `find()`  
   C) `parse()`  
   D) `download()`  

---

## 11차시: 고급 데이터 전처리 및 피처 엔지니어링

### 실습 코드 예제

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

# 예제 데이터
data = {
    'color': ['red', 'blue', 'green', 'blue', 'red'],
    'size': [10, 20, 15, 10, 25],
    'price': [100, 150, 120, 130, 180]
}
df = pd.DataFrame(data)

# 원-핫 인코딩
encoder = OneHotEncoder(sparse=False)
color_encoded = encoder.fit_transform(df[['color']])
print("One-hot encoded:\n", color_encoded)

# 스케일링
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['size', 'price']])
print("Scaled features:\n", df_scaled)

# PCA로 차원 축소
pca = PCA(n_components=1)
pca_feature = pca.fit_transform(df_scaled)
print("PCA 결과:\n", pca_feature)
```

---

### 참고문헌 및 추가 자료

- «Feature Engineering for Machine Learning», Alice Zheng  
- Scikit-learn 공식 문서: https://scikit-learn.org/stable/  
- PCA 튜토리얼: https://arxiv.org/pdf/1404.1100.pdf  

---

### 퀴즈 문제

1. OneHotEncoder가 생성하는 데이터 형태는?  
   A) 희소행렬  
   B) 밀집행렬  
   C) 리스트  
   D) 문자열  

2. PCA의 목적은?  
   A) 데이터 차원 축소  
   B) 데이터 클러스터링  
   C) 결측치 보정  
   D) 과적합 방지  

---

## 12차시: 머신러닝 기초: 지도학습

### 실습 코드 예제

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 로딩
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    test_size=0.3, random_state=42)

# 모델 학습
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```

---

### 참고문헌 및 추가 자료

- «Pattern Recognition and Machine Learning», Christopher M. Bishop  
- Scikit-learn 지도학습 문서: https://scikit-learn.org/stable/supervised_learning.html  

---

### 퀴즈 문제

1. 로지스틱 회귀에서 출력값은 무엇으로 해석되는가?  
   A) 클래스 확률  
   B) 결정 트리 노드  
   C) 회귀의 예측값  
   D) 군집 중심  

2. train_test_split 함수의 작용은?  
   A) 데이터 분류  
   B) 데이터셋 분할  
   C) 모델 평가  
   D) 피처 스케일링  

---

## 13차시: 머신러닝 기초: 비지도학습 및 평가

### 실습 코드 예제

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 데이터 생성
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# KMeans 클러스터링
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# 시각화
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title("KMeans Clustering")
plt.show()
```

---

### 참고문헌 및 추가 자료

- «Unsupervised Learning: Foundations of Neural Computation», Hinton & Dayan  
- Scikit-learn 클러스터링 튜토리얼: https://scikit-learn.org/stable/modules/clustering.html  

---

### 퀴즈 문제

1. K-Means 군집화에서 ‘클러스터 중심’은 무엇을 의미하는가?  
   A) 가장 멀리 떨어진 점  
   B) 군집 내 점들의 평균 위치  
   C) 군집 경계점  
   D) 무작위 위치  

2. 실루엣 점수(Silhouette Score)는 무엇을 평가하는가?  
   A) 클러스터링 품질  
   B) 회귀 정확도  
   C) 분류 손실  
   D) 모델 속도  

---

## 14차시: 딥러닝과 신경망의 이해

### 실습 코드 예제

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 간단한 퍼셉트론 예제
inputs = np.array([0.5, 0.3])
weights = np.array([0.4, 0.7])
bias = 0.1

linear_combination = np.dot(inputs, weights) + bias
output = sigmoid(linear_combination)

print(f"퍼셉트론 출력: {output}")
```

---

### 참고문헌 및 추가 자료

- «Deep Learning», Ian Goodfellow et al.  
- Neural Networks and Deep Learning 강의, Michael Nielsen (http://neuralnetworksanddeeplearning.com/)  

---

### 퀴즈 문제

1. 활성화 함수 Sigmoid의 출력값 범위는?  
   A) 0에서 1  
   B) -1에서 1  
   C) 0에서 무한대  
   D) -무한대에서 무한대  

2. 퍼셉트론이 해결할 수 없는 문제는?  
   A) AND  
   B) OR  
   C) XOR  
   D) 이것 모두 해결 가능  

---

## 15차시: 딥러닝 프레임워크 활용 (PyTorch/TensorFlow)

### 실습 코드 예제 (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 간단한 1층 신경망 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 3)

    def forward(self, x):
        x = self.fc1(x)
        return x

model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 임의 데이터
inputs = torch.randn(5, 4)
targets = torch.randn(5, 3)

# 학습 루프 1회
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()

print("Loss:", loss.item())
```

---

### 참고문헌 및 추가 자료

- PyTorch 공식 튜토리얼: https://pytorch.org/tutorials/  
- TensorFlow 공식 가이드: https://www.tensorflow.org/tutorials  
- «Deep Learning with PyTorch», Eli Stevens et al.

---

### 퀴즈 문제

1. PyTorch에서 `optimizer.zero_grad()`가 필요한 이유는?  
   A) 메모리를 할당하기 위해  
   B) 이전에 계산된 기울기(gradient)를 초기화하기 위해  
   C) 모델 파라미터를 저장하기 위해  
   D) 학습률을 설정하기 위해  

2. `nn.MSELoss()`는 어떤 과제를 위한 손실 함수인가?  
   A) 분류  
   B) 회귀  
   C) 클러스터링  
   D) 강화학습  

---

## 16차시: 합성곱 신경망 (CNN)

### 실습 코드 예제 (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 15 * 15, 10)  # CIFAR-10 이미지 크기 기준

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 15 * 15)
        x = self.fc1(x)
        return x

model = SimpleCNN()
print(model)
```

---

### 참고문헌 및 추가 자료

- Stanford CS231n 강의: http://cs231n.stanford.edu/  
- «Deep Learning», Ian Goodfellow - CNN 챕터 강독  

---

### 퀴즈 문제

1. 합성곱 신경망에서 풀링(pooling) 레이어의 주된 목적은?  
   A) 과적합 방지  
   B) 특징 추출  
   C) 공간 차원 축소  
   D) 학습 속도 증가  

2. 컨볼루션 레이어에서 ‘커널(kernel)’은 무엇인가?  
   A) 손실 함수  
   B) 필터 혹은 가중치 행렬  
   C) 활성화 함수  
   D) 옵티마이저  

---

## 17차시: 순환 신경망 (RNN)

### 실습 코드 예제 (PyTorch)

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

model = SimpleRNN(input_size=10, hidden_size=20, output_size=1)
print(model)
```

---

### 참고문헌 및 추가 자료

- «Deep Learning», Ian Goodfellow: RNN 장  
- PyTorch RNN 튜토리얼: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html  

---

### 퀴즈 문제

1. RNN에서 ‘배치 첫 번째(batch_first=True)’ 옵션은 무엇을 의미하는가?  
   A) 모든 배치를 한 번에 처리함  
   B) 입력 데이터의 첫 번째 차원이 배치 크기임  
   C) 첫 번째 은닉층 활성화 함수를 의미  
   D) 배치 학습을 위한 옵티마이저  

2. LSTM이 RNN보다 우수한 점은?  
   A) 연산 속도가 빠르다  
   B) 장기 의존성 문제를 해결한다  
   C) 파라미터가 적다  
   D) 과적합 방지 기능이 있다  

---

## 18차시: 자연어 처리(NLP) 기초

### 실습 코드 예제

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "I love this product",
    "This is the worst experience",
    "I am very happy",
    "I hate this",
]

labels = [1, 0, 1, 0]  # 1: 긍정, 0: 부정

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

sample = ["I love it", "I don't like it"]
X_sample = vectorizer.transform(sample)
print(model.predict(X_sample))
```

---

### 참고문헌 및 추가 자료

- Jurafsky & Martin, «Speech and Language Processing»  
- «Natural Language Processing with Python», Bird et al.  
- Scikit-learn 텍스트 처리: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction  

---

### 퀴즈 문제

1. CountVectorizer가 생성하는 것은 무엇인가?  
   A) 토큰 리스트  
   B) 단어 문서 행렬 (Bag of Words)  
   C) 임베딩 벡터  
   D) 문장 분리  

2. 감성분석에서 ‘0’과 ‘1’은 보통 무엇을 나타내나요?  
   A) 스팸·정상 구분  
   B) 부정·긍정 분류  
   C) 주제 분류  
   D) 문서 길이  

---

## 19차시: 생성 모델의 원리: Autoencoder

### 실습 코드 예제 (PyTorch)

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
print(model)
```

---

### 참고문헌 및 추가 자료

- «Representation Learning: A Review and New Perspectives», Bengio et al.  
- PyTorch Autoencoder 튜토리얼: https://pytorch.org/tutorials/beginner/autoencoder_tutorial.html  

---

### 퀴즈 문제

1. 오토인코더의 주요 목적은?  
   A) 분류  
   B) 클러스터링  
   C) 데이터 압축 및 특징 추출  
   D) 강화학습  

2. 왜 디코더는 출력에 Sigmoid를 사용하는가?  
   A) 출력값을 확률로 만들기 위해  
   B) 계산 속도 향상을 위해  
   C) 활성화 함수 중 가장 단순해서  
   D) 입력을 재생성하기 위해  

---

## 20차시: 생성적 적대 신경망 (GAN)

### 실습 코드 예제 (PyTorch)

```python
# GAN 모델은 복잡하여 실습 코드는 아래 링크 참조 권장
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py

print("실습 자료는 위 링크 예제 코드를 참고하세요.")
```

---

### 참고문헌 및 추가 자료

- «Generative Adversarial Nets», Goodfellow et al., 2014  
- PyTorch DCGAN 튜토리얼: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html  

---

### 퀴즈 문제

1. GAN에서 Generator의 목적은?  
   A) 진짜 데이터를 판별하는 것  
   B) 가짜 데이터를 생성하는 것  
   C) 손실 함수를 계산하는 것  
   D) 학습률을 조절하는 것  

2. GAN에서 Discriminator는?  
   A) 생성된 데이터와 실제 데이터를 구분한다  
   B) 모델 파라미터를 업데이트한다  
   C) 노이즈를 생성한다  
   D) 활성화 함수를 결정한다  

---

## 21차시: Transformer와 Self-Attention 메커니즘

### 실습 코드 예제 (PyTorch - 간단 Self-Attention)

```python
import torch
import torch.nn.functional as F

def self_attention(q, k, v):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output, attn

# 임의 텐서 생성: (batch_size=1, seq_len=3, d_model=4)
q = torch.rand(1, 3, 4)
k = torch.rand(1, 3, 4)
v = torch.rand(1, 3, 4)

output, attn = self_attention(q, k, v)
print("Self-Attention output:", output)
```

---

### 참고문헌 및 추가 자료

- «Attention Is All You Need», Vaswani et al., 2017  
- Illustrated Transformer: http://jalammar.github.io/illustrated-transformer/  
- PyTorch Transformer Tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html  

---

### 퀴즈 문제

1. Self-Attention에서 Query, Key, Value의 역할은?  
   A) Query와 Key로 가중치 계산, Value와 곱해 결과 생성  
   B) 모두 같은 역할  
   C) 각각 다른 신경망 층  
   D) 모델 초기 하이퍼파라미터  

2. Transformer 모델의 주요 장점은?  
   A) 병렬 처리 가능성  
   B) 낮은 메모리 사용  
   C) 저렴한 계산 비용  
   D) 단일 레이어 구조  

---

