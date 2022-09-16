import fasttext

def train(): # 训练模型
    model = fasttext.train_supervised("fasttext_train.txt", lr=0.1, dim=100,
             epoch=5, word_ngrams=2, loss='softmax')
    model.save_model("model_file_v2.bin")

def test(): # 预测
    classifier = fasttext.load_model("model_file.bin")
    result = classifier.test("fasttext_test.txt")
    print("准确率:", result)
    with open('fasttext_test.txt', encoding='utf-8') as fp:
        for line in fp.readlines():
            line = line.strip()
            if line == '':
                continue
            print(line, classifier.predict([line])[0][0][0])

if __name__ == '__main__':
    #train()
    test()