import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns       # 그림 교차 그리기
import numpy as np

ctx = 'C:/Users/ezen/PycharmProjects/day_02/titanic/data/'
train = pd.read_csv(ctx+'train.csv')                                # Kaggle data 가져 옴
test = pd.read_csv(ctx+'test.csv')
#df = pd.DataFrame(train)
#print(df.columns)
#print(train.head())

''' 
# 아래를 feature 라 함
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

PassengerId                                             # 고객 아이디  
Survived    Survival    0 = No, 1 = Yes                 # 생존여부
pclass    Ticket class    1 = 1st, 2 = 2nd, 3 = 3rd     # 승선권 클래스
sex    Sex                                              # 성별
Age    Age in years                                     # 연령
sibsp    # of siblings / spouses aboard the Titanic     # 동반한 형제자매, 배우자 수
parch    # of parents / children aboard the Titanic     # 동반한 부모, 자식 수
ticket    Ticket number                                 # 티켓 번호
fare    Passenger fare                                  # 티켓의 요금
cabin    Cabin number                                   # 객실 번호
embarked    Port of Embarkation    
C = Cherbourg, Q = Queenstown, S = Southampton          # 승선한 항구명

############# 생존율 #############
f, ax = plt.subplots(1, 2, figsize =(18, 8))
train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)

ax[0].set_title('Survived')
ax[0].set_ylabel('')

sns.countplot('Survived', data=train, ax=ax[1])
ax[1].set_title('Survived')
#plt.show()
# live rate 38.4%

## 데이터는 훈련데이터(train.csv), 목적데이터(test.csd) 두가지로 제공됨. 목적 데이터는 위 항목에서는 survied 정보가
## 빠짐. 이 것이 답이기 때문

#############  성별 생존율 #############
f, ax = plt.subplots(1, 2, figsize =(18, 8))
train['Survived'][train['Sex']=='male'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
train['Survived'][train['Sex']=='female'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[1], shadow=True)
ax[0].set_title('Survived(Male)')
ax[1].set_title('Survived(Female)')
#plt.show()
# 남성 ; 18.9 생존
# 여성 : 74.2% 생존

############# 승선권 vs 생존율 #############
df_1 =  [train['Sex'],train['Survived']]
df_2 =  [train['Pclass']]
df = pd.crosstab(df_1, df_2, margins=True)
#print(df.head())


Pclass             1    2    3  All
Sex    Survived                    
female 0           3    6   72   81
       1          91   70   72  233
male   0          77   91  300  468
       1          45   17   47  109
All              216  184  491  891


############# 승선권 vs 생존율 #############
f, ax = plt.subplots(2, 2, figsize =(20, 15))           # size raise 2*2

sns.countplot('Embarked', data=train, ax=ax[0,0])
ax[0,0].set_title('No. of Passengers Boarded')

sns.countplot('Embarked', hue='Sex', data=train, ax=ax[0,1])
ax[0,1].set_title('Male-Female for Embarked')

sns.countplot('Embarked', hue='Sex', data=train, ax=ax[1,0])
ax[1,0].set_title('Pclass vs Survived')

sns.countplot('Pclass', data=train, ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')

#plt.show()

############# 결측치 #############

#train.info()

RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object


#print(train.isnull().sum())  # null 개수

PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177  # null이지만 버릴 수 없는 수치
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2  # 임의 지점에 넣어 계산해도 무방

'''
sns.set()
def bar_chart(feature):                                                 # 규칙에 따라 패턴을 만들어 동작하는 함수
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['survived', 'dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))
    plt.show()

# bar_chart('Sex')
# bar_chart('Pclass') # 승선권 클래스. 사망한 사람은 3등석, 생존한 사람은 1등석이 많음
# bar_chart('SibSp') # 동반한 형제자매, 배우자 수
# bar_chart('Parch') # 동반한 부모, 자식 수
# bar_chart('Embarked') # 승선한 항구명. S, Q 에 탑승한 사람이 더 많이 사망했고 C 는 덜 사망했다

# Cabin, Ticket 값 삭제
# train = train.drop(['Cabin'], axis = 1)
# test = test.drop(['Cabin'], axis = 1)
# train = train.drop(['Ticket'], axis = 1)
# test = test.drop(['Ticket'], axis = 1)

# print(train.head())   # 다운 방지
# print(test.head())

# Embarked 값 가공
# s_city = train[train['Embarked']=='S'].shape[0]         # 0 means value 1, 1 means array
# c_city = train[train['Embarked']=='C'].shape[0]
# q_city = train[train['Embarked']=='Q'].shape[0]
#
# print('S={}, C={}, Q ={}'.format(s_city, c_city, q_city))
'''
# S=644, C=168, Q =77
'''
# Null data 가공
# train = train.fillna({'Embarked':'S'})              # na means null. 상황 판단하여 처리
# city_mapping =  {'S':1, 'C':2, 'Q':3}       # process needed number
# train['Embarked'] = train['Embarked'].map(city_mapping)
# test['Embarked'] = test['Embarked'].map(city_mapping)  # map method로 문자를 숫자로 변환
#
# print(train.head())
# print(test.head())
'''
#    PassengerId  Survived  Pclass    ...        Fare Cabin  Embarked
# 0            1         0       3    ...      7.2500   NaN         1
# 1            2         1       1    ...     71.2833   C85         2
# 2            3         1       3    ...      7.9250   NaN         1
# 3            4         1       1    ...     53.1000  C123         1
# 4            5         0       3    ...      8.0500   NaN         1
# 
# [5 rows x 12 columns]
#    PassengerId  Pclass   ...    Cabin Embarked
# 0          892       3   ...      NaN        3
# 1          893       3   ...      NaN        1
# 2          894       2   ...      NaN        3
# 3          895       3   ...      NaN        1
# 4          896       3   ...      NaN        1
# 
# [5 rows x 11 columns]
'''
#
combine = [train, test]                 # DataFrame 결합. <class 'pandas.core.frame.DataFrame'>
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)   # Mr. example
# print('dataset', dataset)
# extract : Extract capture groups in the regex pat as columns in a DataFrame.
# k = pd.crosstab(train['Title'],train['Sex'])
# print(k)
'''
Sex       female  male
Title                 
Capt           0     1
Col            0     2
Countess       1     0
Don            0     1
Dr             1     6
Jonkheer       0     1
Lady           1     0
Major          0     2
Master         0    40
Miss         182     0
Mlle           2     0
Mme            1     0
Mr             0   517
Mrs          125     0
Ms             1     0
Rev            0     6
Sir            0     1
'''

# 그룹핑
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess','Lady','Sir'],'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

s= train[['Title','Survived']].groupby(['Title'], as_index=False).mean()
# print(s)

'''
    Title  Survived
0  Master  0.575000
1    Miss  0.702703
2      Mr  0.156673
3     Mrs  0.793651
4    Rare  0.285714
5   Royal  1.000000
'''

train = train.drop(['Name', 'PassengerId'], axis =1)    # 연산에 의미없는 DATA GET RID
test = test.drop(['Name', 'PassengerId'], axis =1)
combine = [train, test]
# print(train.head())

sex_mapping = {'male':0, 'female':1}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Royal':5,'Rare':6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0) # fillna
# print(train.head())

# Age 가공하기

train['Age'] = train['Age'].fillna(-0.5)        # -0.5를 강제 부여하여 bins에서 Unknown으로 처리토록
test['Age'] = test['Age'].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']
train['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)
test['AgeGroup'] = pd.cut(test['Age'], bins, labels = labels)

# print(train.head())

age_title_mapping = {1: "Young Adult", 2:"Student", 3:"Adult", 4:"Baby", 5:"Adult", 6:"Adult"}
for x in range(len(train['AgeGroup'])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train['Title'][x]]
for x in range(len(test['AgeGroup'])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test['Title'][x]]

age_mapping = {'Baby': 1, 'Child':2, 'Teenager':3, 'Student': 4, 'Young Adult':5,'Adult':6, 'Senior':7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = train['AgeGroup'].map(age_mapping)
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)

# print(train.head())

# Fare 처리

train['FareBand'] = pd.qcut(train['Fare'], 4, labels = {1,2,3,4})
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = {1,2,3,4})

train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)
print(train.head())