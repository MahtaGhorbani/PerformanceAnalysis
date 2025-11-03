from numpy import load

data = load('train_5_gram_CJEJUNI.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])
    print(data[item].shape)
