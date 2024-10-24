```
编写人：李鼎铭
名称：生成PPT与智能切割文———代码说明
```

# Generate_PPT_divide_speech

**第一**！进入项目目录，安装依赖：

```
pip install -r requirements.txt
```

本项目分为两个文件目录，分别为：`paper_text`和`project_v3`，分别讲述各自作用。

**注意**：以下生成内容都是自己手动把prompt放到现有的大模型中，也可以自己调用本地模型或者用模型api，利用简单的代码把我下面的文件串起来就好，还是比较简单的。

## 1、文件目录：paper_text

​	本文件目录就是用来存放在操作流程中生成的PPT文件，供给`project_v3`中的代码`2_ppt2txt.py`使用，以获取ppt中每一页的文字。

## 2、文件目录：project_v3

### 文件一：1_text_to_summary_prompt.txt

​	存放**投入文件生成总结**的prompt，将其与论文文件都投入给Kimi.ai，得到PPT生成大纲。

​	之后，将这个PPT生成大纲投喂给Gamma大模型，生成PPT文件，将PPT文件放入文件目录`paper_text`下。

### 文件二：2_ppt2txt.py

**作用：**

​	1、读取**PPT文件**，确保在`paper_text`目录中有该PPT文件，获取PPT每页中的文字，并且放入`ppt.txt`。在代码的main函数中，修改ppt_file这个变量即可。

```python
if __name__ == '__main__':
    # 使用示例
    ppt_file = "../paper_test/AI.pptx"
    word_file = "ppt.txt"
	# 获取PPT每页中的文字，写入ppt.txt文件
    ppt_to_word(ppt_file, word_file)
    with open(word_file, "r", encoding='UTF-8') as f:
        data = f.read().replace('<!!>',' ')
	#写入到4_speech_prompt.txt文件，其存放演讲文稿的prompt
    with open("4_speech_prompt.txt", "w", encoding='UTF-8') as f:
        f.write(generate_speech_prompt(data))
```

​	读取`generate_speech_prompt.txt`这个生成文稿的初始prompt文件，与PPT每页中的文字结合，结合之后写入到`4_speech_prompt.txt`文件，此文件存放需要投入Kimi.ai获取演讲文稿的prompt。

### 文件三：3_divide.py

**作用：**

​	此为核心代码。作用就是实现文稿对应PPT的切分，划分ppt板块与数字人板块。实现流程即基于如下算法：

![](https://img.picui.cn/free/2024/10/14/670d1527d18b2.png)

​	现在开始介绍代码流程：

1、获取文稿中的每一个句子，以“。”、“？”、“！”进行切割，获取句子列表。

```python
def get_speech_sentence():
    with open("5_speech.txt", "r", encoding='UTF-8') as f:
        data = f.read()
    data = data.replace('\n','')
    data = re.split(r'(?<=[。！？])\s*', data)
    return data
```

2、返回一个大列表，包含对应PPT页数的小列表，小列表中存放PPT中的小句子，即返回一个二维列表。

```python
def get_ppt_page():
    with open("ppt.txt", "r", encoding='UTF-8') as f:
        data = f.read().replace('\n','')
        data = data.split('///')
    new_sentence = []
    for i in data:
        sentence = i.split('<!!>')
        sentence = list(filter(bool, sentence))
        if sentence:
            new_sentence.append(merge_2less(sentence))
    return new_sentence
```

3、文本相似度匹配，计算当前文稿句子与PPT中每句话的相似度，返回最大值，由于是向量计算，所以成本较低。

```python
def get_similarity(speech,ppt):
    # 分词
    input_texts = []
    input_texts = ppt.copy()
    input_texts.insert(0, speech)
    batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
    # 获取模型输出
    outputs = model(**batch_dict)
    # 获取嵌入向量
    dimension = 768  # 输出嵌入的维度，应在 [128, 768] 之间
    embeddings = outputs.last_hidden_state[:, 0][:dimension]
    # 归一化嵌入向量
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:1] @ embeddings[1:].T)
    similarity = np.max(scores.tolist())
    print(f"{speech}------与{ppt}------文本相似度: {similarity}\n-------------------------------------------------\n")
    input_texts.clear()
    return similarity
```

4、**核心函数**，根据所写的算法分割文稿，每一if句子在下方代码处已经加上了注释，说明为什么这么做。

```python
def filter_speech_by_ppt(T):
    # 获取每个句子，list[]
    speech_sentences = get_speech_sentence()
    # 获取每个PPT页，list[][]
    ppt_pages = get_ppt_page()
    # excel的字典，存放最终输出,{}
    excel_dic = {}

    # 获取文稿与PPT页各自长度
    speech_len = len(speech_sentences)
    ppt_len = len(ppt_pages)
    print(f"文稿长度：{speech_len}PPT页长度：{ppt_len}\n-------------------------------------------------\n")

    # 设定num用于循环
    speech_num = 0
    ppt_num = 0  # 同时用于循环 与 字典序号存储

    # 数字人序号
    portrait_num = 0

    # 循环条件
    flag = True
    
    while flag:
        # 文稿和PPT都读完才结束
        if speech_num == speech_len-1 and ppt_num == ppt_len-1:
            print(f"\n-------------------------------------------------\n文稿输出长度：{speech_num+1}PPT输出长度：{ppt_num+1}\n-------------------------------------------------\n")
            break

        # 针对一开始文稿与PPT相关度弱的特殊情况
        if ppt_num == 0 and speech_num == 0:
            if get_similarity(speech_sentences[speech_num],ppt_pages[0]) <= T:
                if f'portrait{portrait_num}' in excel_dic:
                    excel_dic[f'portrait{portrait_num}'] += ' ' + speech_sentences[speech_num]
                else:
                    excel_dic.update({f'portrait{portrait_num}': speech_sentences[speech_num]})
                    speech_num += 1

        # 计算similarity，单独拿出来防止冗余计算
        similarity = get_similarity(speech_sentences[speech_num], ppt_pages[ppt_num])

        # 如果 similarity > T ，如果是第一次遇到，放入以ppt+下标放入字典，并且数字人portrait的下标加一，这样遇到弱项关就是新的数字人的键值对。
        # 如果不是第一次遇到，那么久直接在对应键的值上拼接字符串即可。之后计算文稿下一句和PPT下一张的相关度与文稿下一句和当前PPT的相关度，
        # 只有当 文稿下一句和PPT下一张强相关，文稿下一句和当前PPT弱项关时，ppt下标加一，进入下一张ppt
        # 此处防止列表越界，加入ppt_num < ppt_len-1判断语句。
        if similarity > T:
            if f'ppt{ppt_num}' in excel_dic:
                excel_dic[f'ppt{ppt_num}'] += ' '+speech_sentences[speech_num]
            else:
                excel_dic.update({f'ppt{ppt_num}': speech_sentences[speech_num]})
                portrait_num += 1

            if ppt_num < ppt_len-1:
                if get_similarity(speech_sentences[speech_num+1], ppt_pages[ppt_num+1]) > T :
                    if get_similarity(speech_sentences[speech_num+1], ppt_pages[ppt_num]) < T:
                        ppt_num += 1
                        portrait_num -= 1
            speech_num += 1

        # 如果 similarity <= T ，如果是第一次遇到，放入以portrait+下标放入字典，并且数字人ppt的下标加一，这样遇到强项关就是新的ppt的键值对。
        if similarity <= T:
            if f'portrait{portrait_num}' in excel_dic:
                excel_dic[f'portrait{portrait_num}'] += ' '+speech_sentences[speech_num]
            else:
                excel_dic.update({f'portrait{portrait_num}': speech_sentences[speech_num]})
                if ppt_num < ppt_len - 1:
                    ppt_num += 1
            speech_num += 1

    return excel_dic
```

5、写入excel表格，这个比较简单，根据想要的格式来就行：

```python
def write_to_excel(excel_dic):
    # 创建一个新的Excel工作簿
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    sheet['A1'] = '页码'
    sheet['B1'] = '对应语言内容'
    for key, value in excel_dic.items():
        if key.startswith("ppt"):
            key = key[:-1] + str(int(key[-1]) + 1)
        if key.startswith("portrait"):
            key = "数字人" + str(int(key[-1]))
        sheet.append([key, value])
    # 保存工作簿
    workbook.save('example.xlsx')
    print("成功写入！")
```

6、主函数，重点就是设定了阈值与embeeding模型，阈值设置为0.75，是我多次看输出结果获得的自认为最佳阈值，总的来说，效果不错。

```python
if __name__ == '__main__':

    # 加载分词器和模型
    model_name_or_path = 'iic/gte_sentence-embedding_multilingual-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)

    # 设定阈值T
    threshold_T = 0.75
    excel_dic = filter_speech_by_ppt(threshold_T)
    write_to_excel(excel_dic)

    # 打印字典
    for key, value in excel_dic.items():
        print(f"{key}: {value}")
        
	# 写入Json文件，便于观察
    # with open("save.json", "w", encoding='utf-8') as f:
    #     f.write(json.dumps(excel_dic,ensure_ascii=False ))
```

### 文件四：4_speech_prompt.txt

​	存放**生成演讲文稿**的prompt，在运行完`2_ppt2txt.py`代码之后，prompt就会生成在这里，复制其中的内容投入给Kimi.ai，得到演讲文稿。并将演讲文稿放入到`5_speech.txt`文本文件中。

### 文件五：5_speech.txt

​	存放**演讲文稿**，手动将Kimi.ai输出的演讲文稿存放到此处。

### 文件六、文件七：generate_speech_prompt.txt、ppt.txt

​	**中间过渡文件**，除非有需求否则不需要操作的文件。

​	`generate_speech_prompt.txt`是生成演讲文稿的初始prompt，也就是放置着一些问题的约束条件，`2_ppt2txt.py`就是读取此文件再把prompt放到`4_speech_prompt.txt`上的。需要改约束条件的时候再在这里进行修改。

​	`ppt.txt`放置ppt每一页的内容的文本文件，只是便于观察。其中句子分隔符“<!!>”是在`2_ppt2txt.py`设置，但是最好不要改，因为在`3_divide.py`中需要根据这个分隔符进行划分句子。