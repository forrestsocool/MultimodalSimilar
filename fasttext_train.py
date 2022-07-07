import fasttext

def train(): # 训练模型
    model = fasttext.train_supervised("fastext_train_data.txt", lr=0.1, dim=100,
             epoch=5, word_ngrams=2, loss='softmax')
    model.save_model("model_file.bin")

def test(): # 预测
    classifier = fasttext.load_model("model_file.bin")
    result = classifier.test("fastext_test_data.txt")
    print("准确率:", result)
    with open('fastext_test_data.txt', encoding='utf-8') as fp:
        for line in fp.readlines():
            line = line.strip()
            if line == '':
                continue
            print(line, classifier.predict([line])[0][0][0])

if __name__ == '__main__':
    train()
    test()