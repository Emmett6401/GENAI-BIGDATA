# PART 3: 심화 및 응용 (Weeks 8-11)

---

## 22차시: 거대 언어 모델(LLM)의 이해

### 실습 코드 예제 (Hugging Face Transformers로 텍스트 생성)

```python
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')
result = generator("Artificial Intelligence is", max_length=30, num_return_sequences=1)
print(result[0]['generated_text'])
```

---

### 참고문헌 및 추가 자료

- «Language Models are Few-Shot Learners», Brown et al., 2020  
- Hugging Face Transformers 공식 문서: https://huggingface.co/transformers/  
- OpenAI API 문서: https://platform.openai.com/docs/api-reference  

---

### 퀴즈 문제

1. GPT 모델은 어떤 유형의 학습 방식을 사용하는가?  
   A) 비지도 학습(Self-supervised Learning)  
   B) 강화 학습  
   C) 지도 학습만  
   D) 전이학습만  

2. 사전학습(Pre-training)과 미세조정(Fine-tuning)의 차이는?  
   A) 사전학습은 넓은 범위 데이터, 미세조정은 특정 업무에 맞춤  
   B) 둘은 동일 용어  
   C) 미세조정은 사전학습 후 실행 중지 단계  
   D) 사전학습은 학습률 조절 과정  

---

## 23차시: LLM 활용 기법: 프롬프트 엔지니어링

### 실습 코드 예제 (OpenAI API 프롬프트 설계)

```python
import openai

openai.api_key = 'YOUR_API_KEY'

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="You are a helpful assistant. Explain the importance of AI ethics.",
    max_tokens=100
)

print(response.choices[0].text.strip())
```

---

### 참고문헌 및 추가 자료

- «Prompt Engineering Guide», https://github.com/dair-ai/Prompt-Engineering-Guide  
- OpenAI GPT 프롬프트 설계 문서  
- «Chain-of-thought Prompting Efficacy», Wei et al., 2022  

---

### 퀴즈 문제

1. 제로샷 학습이란?  
   A) 학습 데이터 없이 바로 답을 생성하는 프롬프트 방식  
   B) 사전학습을 반복하는 방법  
   C) 소량 데이터로 학습하는 방식  
   D) 모델을 미세조정하는 기법  

2. Role Prompting 기법은 무엇을 지정하는가?  
   A) 모델의 응답 스타일과 역할  
   B) 데이터 저장소 위치  
   C) 인퍼런스 속도  
   D) GPU 자원 

---

## 24차시: LLM 활용 기법: RAG와 파인튜닝

### 실습 코드 예제 (FAISS 벡터 DB 이용한 문서 검색)

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

docs = ["AI is transforming the world.", "Climate change is an urgent issue.", "Health care advances rapidly."]
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

query = "Tell me about AI."
query_emb = model.encode([query])
D, I = index.search(np.array(query_emb), k=1)
print("Closest document:", docs[I[0][0]])
```

---

### 참고문헌 및 추가 자료

- «Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks», Lewis et al., 2020  
- FAISS 공식 문서: https://faiss.ai/  
- Sentence Transformers: https://www.sbert.net/  

---

### 퀴즈 문제

1. RAG에서 ‘검색 증강’이란?  
   A) 외부 데이터베이스에서 관련 문서 검색 후 생성에 활용  
   B) 모델 학습 데이터 증강  
   C) 텍스트 분류 향상 기법  
   D) GPU 가속 기술  

2. 파인튜닝(Fine-tuning) 시도의 주요 목적은?  
   A) 특정 도메인에 모델 맞춤  
   B) 기본 모델 크기 축소  
   C) 클라우드 배포  
   D) 데이터 전처리  

---

## 25차시: 멀티모달 생성 모델: Text-to-Image

### 실습 코드 예제 (Stable Diffusion API 호출 예제)

```python
# 예제는 Hugging Face API, 토큰 필요
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a futuristic cityscape at sunset"
image = pipe(prompt).images[0]
image.save("output.png")
```

---

### 참고문헌 및 추가 자료

- «High-Resolution Image Synthesis with Latent Diffusion Models», Rombach et al., 2022  
- Stable Diffusion 허깅페이스 문서: https://huggingface.co/CompVis/stable-diffusion-v1-4  

---

### 퀴즈 문제

1. 확산 모델(Diffusion Model)의 핵심 동작은?  
   A) 점진적으로 노이즈를 제거하며 이미지 생성  
   B) GAN과 동일 작동  
   C) 텍스트만 분석  
   D) 단순히 이미지 리사이징  

2. CLIP 모델은 무엇을 연결하는가?  
   A) 텍스트와 이미지 임베딩  
   B) 오디오 및 텍스트  
   C) 이미지와 비디오  
   D) 데이터베이스와 AI  

---

## 26차시: 멀티모달 생성 모델: 그 외 응용

### 실습 코드 예제 (이미지 캡셔닝 간단 구현)

```python
# PyTorch 이미지-텍스트 모델 예제는 코드 길어 별도 자료 참고 권장
print("이미지 캡셔닝 간단 구현은 Hugging Face 모델 활용을 추천합니다.")
```

---

### 참고문헌 및 추가 자료

- «Show, Attend and Tell: Neural Image Caption Generation with Visual Attention», Xu et al., 2015  
- Hugging Face 이미지 캡셔닝 모델: https://huggingface.co/models?pipeline_tag=image-captioning  

---

### 퀴즈 문제

1. 이미지 캡셔닝 모델의 기본 구조는?  
   A) Encoder-Decoder  
   B) GAN  
   C) Random Forest  
   D) PCA  

2. 멀티모달 임베딩의 장점은?  
   A) 다양한 데이터 유형을 통합하여 이해 가능  
   B) 컴퓨팅 자원 절약  
   C) 전통 머신러닝 대체  
   D) 디버깅 간편성  

---

## 27차시: 빅데이터 생태계의 이해

### 실습 코드 예제

```bash
# Hadoop HDFS 기본 명령어 실습 예시
hdfs dfs -mkdir /user/username/input
hdfs dfs -put ./localfile.txt /user/username/input/
hdfs dfs -ls /user/username/input
```

---

### 참고문헌 및 추가 자료

- «Hadoop: The Definitive Guide», Tom White  
- Apache Hadoop 공식문서: https://hadoop.apache.org/docs/  

---

### 퀴즈 문제

1. HDFS란?  
   A) 분산 파일 시스템  
   B) SQL 데이터베이스  
   C) 머신러닝 라이브러리  
   D) API 통신 규약  

2. MapReduce의 역할은?  
   A) 대용량 데이터 병렬 처리  
   B) UI 생성  
   C) 데이터 시각화  
   D) 모델 학습  

---

## 28차시: 대용량 데이터 처리: Apache Spark

### 실습 코드 예제 (PySpark)

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Example").getOrCreate()
df = spark.read.csv("data.csv", header=True, inferSchema=True)
df.show(5)
filtered_df = df.filter(df['age'] > 30)
filtered_df.show()
```

---

### 참고문헌 및 추가 자료

- «Learning Spark», Holden Karau et al.  
- Apache Spark 공식 문서: https://spark.apache.org/docs/latest/  

---

### 퀴즈 문제

1. Apache Spark의 Driver 역할은?  
   A) 작업 제어 및 스케줄링  
   B) 클러스터 노드 역할  
   C) 데이터 저장  
   D) 모델 평가  

2. RDD와 DataFrame의 차이점은?  
   A) RDD는 낮은 수준, DataFrame은 구조화 데이터 처리  
   B) 둘은 동일하다  
   C) RDD는 SQL 전용  
   D) DataFrame은 비지도학습 전용  

---

## 29차시: Spark를 활용한 머신러닝

### 실습 코드 예제

```python
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("MLlibExample").getOrCreate()
data = [(0, 1.0, 2.0), (1, 1.5, 1.8), (2, 5.0, 8.0), (3, 8.0, 8.0)]
df = spark.createDataFrame(data, ["id", "feature1", "feature2"])
assembler = VectorAssembler(inputCols=["feature1", "feature2"], outputCol="features")
dataset = assembler.transform(df)
kmeans = KMeans(k=2, seed=1, featuresCol='features')
model = kmeans.fit(dataset)
predictions = model.transform(dataset)
predictions.show()
```

---

### 참고문헌 및 추가 자료

- Apache Spark MLlib 문서: https://spark.apache.org/mllib/  
- «Advanced Analytics with Spark», Sandy Ryza et al.  

---

### 퀴즈 문제

1. Spark MLlib에서 VectorAssembler의 역할은?  
   A) 여러 컬럼을 하나의 벡터 컬럼으로 변환  
   B) 모델 학습  
   C) 데이터 시각화  
   D) 클러스터링 평가  

2. Spark MLlib에서 KMeans는 어떤 유형의 알고리즘인가?  
   A) 지도학습  
   B) 비지도 학습  
   C) 강화학습  
   D) 전이학습  

---

## 30차시: 실시간 데이터 처리와 AI

### 실습 코드 예제

```python
# Kafka Producer 예제 (Python)
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers='localhost:9092')
producer.send('test_topic', b'Hello, Kafka!')
producer.flush()
```

---

### 참고문헌 및 추가 자료

- Apache Kafka 공식 문서: https://kafka.apache.org/documentation/  
- Spark Structured Streaming: https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html  

---

### 퀴즈 문제

1. Kafka에서 Producer의 역할은?  
   A) 메시지 발행자  
   B) 메시지 소비자  
   C) 데이터 저장소  
   D) 토픽 관리  

2. 실시간 스트림 처리와 배치 처리의 차이는?  
   A) 스트림 처리는 연속적 데이터, 배치는 고정 데이터 처리  
   B) 모두 동일  
   C) 배치는 빠름  
   D) 스트림 처리는 비효율적  

---

## 31차시: MLOps: 모델 배포 및 서빙

### 실습 코드 예제 (FastAPI)

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    x: float
    y: float

@app.post("/predict")
def predict(data: InputData):
    result = data.x + data.y
    return {"result": result}

# 실행: uvicorn main:app --reload
```

---

### 참고문헌 및 추가 자료

- FastAPI 공식문서: https://fastapi.tiangolo.com/  
- «Machine Learning Engineering», Andriy Burkov  

---

### 퀴즈 문제

1. FastAPI에서 Endpoint를 정의하는 데 사용하는 데코레이터는?  
   A) @app.route  
   B) @app.post/@app.get  
   C) @endpoint  
   D) @function  

2. MLOps가 중요한 이유는?  
   A) 모델 재현과 운영 자동화  
   B) 모델 학습 속도 향상  
   C) 데이터 수집  
   D) 단순 UI 제작  

---

## 32차시: MLOps: CI/CD/CT 및 모니터링

### 실습 코드 예제

```yaml
# GitHub Actions 간단 워크플로우 예시 (.github/workflows/ci.yml)
name: CI

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest
```

---

### 참고문헌 및 추가 자료

- GitHub Actions 문서: https://docs.github.com/en/actions  
- «Introducing MLOps», Microsoft Azure Docs  

---

### 퀴즈 문제

1. CI/CD에서 ‘CI’는 무엇의 약자인가?  
   A) Continuous Integration  
   B) Continuous Improvement  
   C) Continuous Innovation  
   D) Continuous Interaction  

2. CT(Continuous Testing)은 무엇을 의미하는가?  
   A) 자동화된 테스트 지속 실행  
   B) 페어 프로그래밍  
   C) 모델 하이퍼파라미터 튜닝  
   D) 정적 코드 분석  

---

## 33차시: MLOps 플랫폼 및 도구

### 실습 코드 예제 (MLflow)

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
model = RandomForestClassifier()
model.fit(X_train, y_train)

mlflow.sklearn.log_model(model, "random_forest_model")
```

---

### 참고문헌 및 추가 자료

- MLflow 공식 문서: https://mlflow.org/  
- Kubeflow 문서: https://www.kubeflow.org/  
- AWS SageMaker 소개: https://aws.amazon.com/sagemaker/  

---

### 퀴즈 문제

1. MLflow에서 ‘Experiment’란?  
   A) 여러 실험 단위를 묶은 관리 단위  
   B) 모델 저장소  
   C) 배포 서버  
   D) 코드 리포지토리  

2. Kubeflow는 주로 어떤 플랫폼에서 사용되는가?  
   A) Kubernetes  
   B) AWS Lambda  
   C) Hadoop  
   D) Docker만  

---

# PART 4: 프로젝트 및 최신 동향 (Weeks 12-13)

---

네! PART 4 각 차시별로도 실습 코드 예제, 참고문헌, 추가 자료, 퀴즈 문제를 제공해 드립니다.

---

## 34차시: 최종 프로젝트 기획

### 실습 코드 예제

```python
# 예) 데이터셋 탐색 및 EDA 기본 실습 (Kaggle Titanic 데이터 사용)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
train = pd.read_csv('train.csv')

# 기초 통계
print(train.describe())

# 생존자 수 시각화
sns.countplot(x='Survived', data=train)
plt.show()

# 성별에 따른 생존율 분석
sns.barplot(x='Sex', y='Survived', data=train)
plt.show()
```

---

### 참고문헌 및 추가 자료

- «Data Science Project Management», Roger D. Peng  
- Kaggle Titanic: https://www.kaggle.com/c/titanic  

---

### 퀴즈 문제

1. 프로젝트 기획 시 가장 먼저 해야 할 일은?  
   A) 데이터 탐색 및 문제 정의  
   B) 모델 학습  
   C) 코드 작성  
   D) 결과 보고서 작성  

---

## 35차시: 최종 프로젝트 중간 점검 및 멘토링

### 실습 코드 예제

- 중간 점검 기간이므로 주로 코드 리뷰 및 수정, 문제 해결 워크숍 진행

---

### 참고자료

- GitHub PR(풀 리퀘스트) 리뷰 가이드  
- 코드 디버깅 및 문제 해결 백서

---

### 퀴즈 문제

1. 중간 점검 회의에서 가장 중요한 내용은?  
   A) 문제점 및 해결 방안 논의  
   B) 새로운 데이터 수집  
   C) 모델 버전 업로드  
   D) 프레젠테이션 제작  

---

## 36차시: 모델 고도화 및 시스템 구현

### 실습 코드 예제

```python
# 하이퍼파라미터 튜닝 예제: GridSearchCV (scikit-learn)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = { 
    'n_estimators': [50, 100],
    'max_depth' : [4, 6, 8]
}

clf = RandomForestClassifier()
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

---

### 참고문헌 및 추가 자료

- Scikit-learn 하이퍼파라미터 튜닝: https://scikit-learn.org/stable/modules/grid_search.html  
- FastAPI 공식문서 (API 연동): https://fastapi.tiangolo.com/  

---

### 퀴즈 문제

1. 하이퍼파라미터 튜닝의 목적은?  
   A) 모델 성능 최적화  
   B) 코드 단순화  
   C) 데이터 정규화  
   D) 데이터 시각화  

---

## 37차시: 최종 프로젝트 발표 및 평가

### 실습 코드 예제

- 발표 자료 작성 및 데모 시연 준비  
- 프로젝트 결과 핵심 지표 시각화 예시 (matplotlib, seaborn 활용)

---

### 참고자료

- 효과적인 데이터 과학 발표법 참고 도서 및 온라인 강좌  
- 피드백 수렴 및 개선 가이드라인  

---

### 퀴즈 문제

1. 발표 평가에서 중요한 항목은?  
   A) 문제 해결력 및 결과의 명확성  
   B) 발표 시간 준수  
   C) 화려한 발표 자료  
   D) 코드 속도  

---

## 38차시: 생성형 AI와 빅데이터의 최신 동향

### 실습 코드 예제

- 최신 논문 요약 보고서 작성 연습 (구글 콜랩 문서 활용 권장)

---

### 참고문헌 및 자료

- 최신 AI 연구 논문 아카이브(ArXiv): https://arxiv.org/list/cs.AI/recent  
- AI 뉴스 및 테크 블로그 (OpenAI, DeepMind, FAIR)  

---

### 퀴즈 문제

1. AI 에이전트란?  
   A) 자율적 의사결정 수행 시스템  
   B) 인간 대체 게임기  
   C) 단순 규칙 기반 봇  
   D) 컴퓨터 하드웨어  

---

## 39차시: AI 윤리와 미래 전망

### 실습 코드 예제

- AI 윤리 관련 사례 토론 및 보고서 작성 실습

---

### 참고문헌 및 추가 자료

- «Weapons of Math Destruction», Cathy O’Neil  
- AI 윤리 가이드라인 (OECD, UNESCO)  

---

### 퀴즈 문제

1. AI 개발 시 편향성 문제를 해결하는 방법은?  
   A) 다양하고 공정한 데이터 수집  
   B) 인공신경망 제거  
   C) 사용자 차단  
   D) 학습 속도 증가  

