import re
import json
import torch.nn.functional as F
from modelscope import AutoModel, AutoTokenizer
import numpy as np
import openpyxl

# v3版本修改了读PPT内容的函数，get_ppt_page()返回一个大列表，包含对应PPT页数的小列表，小列表中存放PPT中的小句子
# similarity函数用来比较speech句子跟一页中每一句进行比较，即返回最大值。


# 获取文稿中的每一个句子
def get_speech_sentence():
    with open("5_speech.txt", "r", encoding='UTF-8') as f:
        data = f.read()
    data = data.replace('\n','')
    data = re.split(r'(?<=[。！？])\s*', data)
    return data


# 将字数过少的句子与下一句结合
def merge_2less(elements):
    merged_elements = []
    flag = False
    for i, element in enumerate(elements):
        if flag == True:
            # 如果当前元素的字数大于2，则直接添加到merged_elements列表中
            merged_elements[-1] += ' ' + element
            flag = False
        else:
            if len(element) <= 10:
                if i == len(elements) - 1:
                    merged_elements[-1] += ' ' + element
                else:
                    merged_elements.append(element)
                    flag = True
            else:
                merged_elements.append(element)
    return merged_elements


# get_ppt_page()返回一个大列表，包含对应PPT页数的小列表，小列表中存放PPT中的小句子
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

# 计算当前文稿句子与PPT中每句话的相似度，返回最大值，由于是向量计算，所以成本较低。
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

# 核心函数，分割文稿
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

    # with open("save.json", "w", encoding='utf-8') as f:
    #     f.write(json.dumps(excel_dic,ensure_ascii=False ))