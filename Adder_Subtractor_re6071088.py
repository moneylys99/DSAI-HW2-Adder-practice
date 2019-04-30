#!/usr/bin/env python
# coding: utf-8

# # DSAI HW2: Adder & Subtractor Practice by LSTM     
# ## Chien, Hsin Yen
# ### RE6071088, Institute of Data Science  

# Data Generation: 80000 data for adder, 80000 data for subtractor  
# Digits available: <= 3  
# LSTM:  
# 1 hidden layer  
# Hidden layer size = 128  
# Batch size = 128  
# training epoch = 100  

# # Import package

# In[1]:


from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import numpy as np
from six.moves import range


# # Parameters Config

# In[2]:


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


# In[3]:


TRAINING_SIZE =160000
DIGITS = 3
REVERSE = False
MAXLEN = DIGITS + 1 + DIGITS
chars = '0123456789+- '
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1


# In[4]:


class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
    
    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x
    
    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[i] for i in x)


# In[5]:


ctable = CharacterTable(chars)


# In[6]:


ctable.indices_char


# # Data Generation

# ### Generating data for adder

# In[7]:


questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < TRAINING_SIZE/2:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    ans += ' ' * (DIGITS + 1 - len(ans))
    if REVERSE:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)


# ### Generating data for subtractor

# In[8]:


while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    q = '{}-{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a - b)
    ans += ' ' * (DIGITS + 1 - len(ans))
    if REVERSE:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total questions:', len(questions))


# In[9]:


print(questions[:3], expected[:3])
print(questions[150000:150003], expected[150000:150003])


# # Processing

# In[10]:


print('Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(expected), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)


# In[11]:


indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# train_test_split
train_x = np.concatenate((x[:20000], x[80000:100000]), axis=0)
train_y = np.concatenate((y[:20000], y[80000:100000]),axis=0)
test_x = np.concatenate((x[20000:80000], x[100000:]),axis=0)
test_y = np.concatenate((y[20000:80000], y[100000:]),axis=0)

split_at = len(train_x) - len(train_x) // 10
(x_train, x_val) = train_x[:split_at], train_x[split_at:]
(y_train, y_val) = train_y[:split_at], train_y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

print('Testing Data:')
print(test_x.shape)
print(test_y.shape)


# In[12]:


print("input: ", x_train[:3], '\n\n', "label: ", y_train[:3])


# # Build Model

# In[13]:


print('Build model...')

# Initialising the RNN
model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
model.add(layers.RepeatVector(DIGITS + 1))
for i in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

model.summary()


# # Training
# Combine adder and substractor  
# Training epoch = 100  

# In[14]:


for iteration in range(100):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)


# # Testing 1 (by test data)

# In[17]:


print("MSG : Prediction")
preds = model.predict_classes(test_x)
for i in range(10):
    q = ctable.decode(test_x[i])
    correct = ctable.decode(test_y[i])
    guess = ctable.decode(preds[i], calc_argmax=False)
    print('Q', q[::-1] if REVERSE else q, end=' ')
    print('T', correct, end=' ')
    if correct == guess:
        print(colors.ok + '☑' + colors.close, end=' ')
    else:
        print(colors.fail + '☒' + colors.close, end=' ')
    print(guess)


# # Testing 2 (by new question)
# Q: 760+172, 529+39 , 227-530, 866+777, 10-879 , 630-342, 235-111, 688+524, 999+166

# In[66]:


newQ = ['760+172', '529+39 ', '227-530', '866+777','10-879 ','630-342', '235-111','688+524', '999+166']
newA = ['932', '568 ', '-303', '1643', '-869', '288', '124', '1212', '1165']
print('Vectorization...')
x = np.zeros((len(newQ), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(newA), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(newQ):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(newA):
    y[i] = ctable.encode(sentence, DIGITS + 1)
    
print("MSG : Prediction")
for i in range(len(newQ)):
    preds = model.predict_classes(x)
    q = ctable.decode(x[i])
    correct = ctable.decode(y[i])
    guess = ctable.decode(preds[i], calc_argmax=False)
    print('Q', q[::-1] if REVERSE else q, end=' ')
    print('T', correct, end=' ')
    if correct == guess:
        print(colors.ok + '☑' + colors.close, end=' ')
    else:
        print(colors.fail + '☒' + colors.close, end=' ')
    print(guess)


# In[ ]:




