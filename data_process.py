import pandas as pd
import numpy as np
# 字形CNN
def save_cnn_inputs(address):
    sign_dict = {}
    with open(address + 'sign_dict.txt', 'r', encoding='utf-8')as f:
        datas = f.readlines()
        for data in datas:
            sign_dict[data.split('\t')[1].rstrip('\n')] = data.split('\t')[0]
    num_to_np = {}
    import cv2
    def cv_imread(filePath):
        return cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)

    for key, value in sign_dict.items():
        pic_data = cv_imread(address + str(int(value) + 1) + '.png').tolist()
        list_data = []
        for raw in pic_data:
            for col in raw:
                list_data.append(col)
        num_to_np[key] = list_data
    none_data = cv_imread(address + '0.png').tolist()
    list_data = []
    for raw in none_data:
        for col in raw:
            list_data.append(col)
    num_to_np[' '] = list_data
    np.save(address + 'data_dict.npy', num_to_np)

def get_cnn_inputs(tokens, max_seq_length, num_to_np, ):
    datas = []
    for token in tokens:
        try:
            datas.append(num_to_np[token])
        except KeyError:
            print(num_to_np)
            print("*** picture not found! ***")
            print(token)
            exit()
    while len(datas) < max_seq_length:
        datas.append(num_to_np[' '])
    assert len(datas) == max_seq_length
    datas = np.reshape(np.array(datas), -1)

    return datas.tolist()

# 五笔GRU
def wubi_input(tokens, max_seq_length):
    file = pd.read_excel('wubi.xlsx')
    char_dict = {}
    for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
        char_dict[c] = i+1
    char_dict[' '] = 0
    wubi_dict = {}
    for row in file.values:
        assert len(row) == 1
        wubi_dict[row[0][0]] = row[0][1:]
    wubi_dict[' '] = ' '
    wubi_dict['[UNK]'] = '    '
    datas = []
    for i, token in enumerate(tokens):
        try:
            chars = wubi_dict[token]
        except KeyError:
            chars = wubi_dict['[UNK]']
            # print(wubi_dict)
            # print("*** word not found! ***")
            # print(tokens[i-10:i+1])
            # exit()
        char_code = []
        for c in chars:
            char_code.append(char_dict[c])
        while len(char_code) < 4:
            char_code.append(0)
        assert len(char_code) == 4
        datas.append(char_code)
    while len(datas) < max_seq_length:
        datas.append([0, 0, 0, 0])
    assert len(datas) == max_seq_length
    datas = np.reshape(np.array(datas), -1)
    return datas.tolist()
if __name__ == '__main__':
    wubi_input()