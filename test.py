import pandas as pd
def chaxunchangdu():
    all_data = []
    sentence = ''
    with open('datas//train_all.txt') as f:
        for i, line in enumerate(f.readlines()):
            if line == '\n':
                all_data.append(sentence)
                sentence = ''
                continue
            sentence += line.split(' ')[0]
        f.close()


    len_num = []
    max_len = 0
    for sen in all_data:
        len_num.append(len(sen))
    print(sorted(len_num)[int(len(len_num)*0.8)])

def wubi():
    file = pd.read_excel('wubi.xlsx')
    wubi_dict = {}
    for row in file.values:
        assert len(row) == 1
        wubi_dict[row[0][0]] = row[0][1:]
    return wubi_dict
if __name__ == '__main__':
    wubi()









































