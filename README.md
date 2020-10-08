# Knowledge-Retrieval
The related papers of Knowledge and Retrieval

-------
## [Content](#content)
1. [Knowledge-Papers](#k-papers)
2. [Retieval-Papers](#d-papers)
3. [Related papers](#related-papers)
4. [Others](#others)

## [Knowledge-Papers](#content)
1. **Chinese NER Using Lattice LSTM** *ACL 2018* [[paper](https://arxiv.org/pdf/1805.02023.pdf) / [code](https://github.com/jiesutd/LatticeLSTM)]


## [Retieval-Papers](#content) 
1. **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** *EMNLP 2019* [[paper](https://arxiv.org/abs/1908.10084) / [code](https://github.com/UKPLab/sentence-transformers)]
<br/><img src="./images/SentenceBERT.jpg" width="300"  alt="model structure"/><br/>

2. **DC-BERT: Decoupling Question and Document for Efficient Contextual Encoding** *SIGIR2020* [[paper](https://arxiv.org/pdf/2002.12591.pdf)]
<br/><img src="./images/DC-BERT.jpg" width="500"  alt="model structure"/><br/>

3. **REALM: Retrieval-Augmented Language Model Pre Training** *arXiv2020* [[paper](https://arxiv.org/abs/2002.08909)]
<br/><img src="./images/REALM.jpg" width="800"  alt="model structure"/><br/>
<br/><img src="./images/REALM02.jpg" width="500"  alt="model structure"/><br/>

4. **Zero-Shot Entity Linking with Dense Entity Retrieval** *arXiv2020* [[paper](https://arxiv.org/abs/1911.03814v2)]

5. **Dense Passage Retrieval for Open-Domain Question Answering** *ACL2020* [[paper](https://arxiv.org/pdf/2004.04906.pdf) / [code](https://fburl.com/qa-dpr)]
- DPR(Dense Passage Retriever) 稠密文章检索

      训练 由于原始的BERT预训练模型很通用，而且得到的句向量是不具备相似性计算要求的，也就是说，相似的两句话输入BERT得到的两个向量并不一定很近。因此，本论文基于Natural Questions (NQ)，TriviaQA (Trivia)，WebQuestions (WQ)，CuratedTREC (TREC)，SQuAD v1.1等数据集重新训练了一个BERT，专门用来生成问题段落的句向量表示。
  
- 数据构造

  **负样本构造方法：**
  
      Random：从语料中随机抽取；
      BM25：使用BM25检索的不包含答案的段落；
      Gold：训练集中其他问题的答案段落
  
  **正样本构造方法：**
  
        因为数据集如TREC, WebQuestions 和 TriviaQA中只包含问题和答案，没有段落文本，因此，论文通过BM25算法，在Wikipedia中进行检索，取top-100的段落，如果答案没有在里面，则丢掉这个问题。对于  SQuAD 和 NQ数据集，同样使用问题在Wikipedia检索，如果检索到的正样本段落能够在数据集中匹配则留下，否则丢掉。

6. **Latent Retrieval for Weakly Supervised Open domain Question Answering** *ACL2019* [[paper](https://arxiv.org/pdf/1906.00300.pdf)]
<br/><img src="./images/DC-BERT.jpg" width="500"  alt="model structure"/><br/>

7. **Learning to Learn from Weak Supervision by Full Supervision** *ICML 2018* [[paper](http://metalearning.ml/2017/papers/metalearn17_dehghani.pdf)]

8. **Selective Weak Supervision for Neural Information Retrieval** *WWW 2020* [[paper](https://arxiv.org/abs/2001.10382)]

## [Related papers](#content)


## [Others](#content)
1. [XLink](https://github.com/solitaryzero/XLink)
2. [OpenMatch](https://github.com/thunlp/OpenMatch)
3. [NeuIRPapers](https://github.com/thunlp/NeuIRPapers)
4. [碎碎念：Transformer的细枝末节](https://zhuanlan.zhihu.com/p/60821628)
5. [BERT模型可以使用无监督的方法做文本相似度任务吗？](https://www.zhihu.com/question/354129879)
