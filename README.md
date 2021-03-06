# Knowledge-Retrieval
The related papers of Knowledge and Retrieval

-------
## [Content](#content)
1. [Knowledge-Papers](#k-papers)
2. [Retieval-Papers](#d-papers)
3. [Related papers](#related-papers)
4. [Competitions](#competitions)
5. [Conference](#conference)
6. [Dataset](#dataset)
7. [Data Augment](#data-augment)
99. [Others](#others)


## [Knowledge-Papers](#content)
1. **Chinese NER Using Lattice LSTM** *ACL 2018* [[paper](https://arxiv.org/pdf/1805.02023.pdf) / [code](https://github.com/jiesutd/LatticeLSTM)]
<br/><img src="./images/Lattice LSTM.jpg" width="350"  alt="model structure"/><br/>
2. **LAVA NAT: A Non-Autoregressive Translation Model with Look-AroundDecoding and Vocabulary Attention** *ICML 2020* [[paper](https://arxiv.org/pdf/2002.03084.pdf)]
<br/><img src="./images/NAT1.jpg" width="300"  alt="model structure"/><img src="./images/NAT2.jpg" width="350"  alt="model structure"/><br/>
<br/><img src="./images/NAT3.jpg" width="600"  alt="model structure"/><br/>

3. **Blank Language Models** *EMNLP 2020* [[paper](https://arxiv.org/pdf/2002.03079.pdf) / [code](https://github.com/Varal7/blank_language_model)]
<br/><img src="./images/BLM.jpg" width="600"  alt="model structure"/><br/>

4. **Spelling Error Correction with Soft-Masked BERT** *ACL 2020* [[paper](https://arxiv.org/pdf/2005.07421.pdf) / [code](https://blog.csdn.net/qq_35128926/article/details/106770581)]

5. **Exploiting Structured Knowledge in Text via Graph-Guided Representation Learning (GLM)** *arxiv2020*
<br/><img src="./images/GLM01.jpg" width="600"  alt="model structure"/><br/>
<br/><img src="./images/GLM02.jpg" width="700"  alt="model structure"/><br/>

6. **基于动态知识选择的预训练模型** *CCKS 2020* \[paper? / [video](https://hub.baai.ac.cn/view/4095) / [PPT](https://hub.baai.ac.cn/view/4155)\]

7. **CoLAKE: Contextualized Language and Knowledge Embedding** *COLING 2020* [[paper](https://arxiv.org/pdf/2010.00309.pdf) / [code](https://github.com/txsun1997/CoLAKE)]
<br/><img src="./images/CoLAKE01.jpg" width="600"  alt="model structure"/><br/>
<br/><img src="./images/CoLAKE02.jpg" width="700"  alt="model structure"/><br/>

8. **JAKET: Joint Pre-training of Knowledge Graph and Language Understanding** *arxiv2020* [[paper](https://arxiv.org/pdf/2010.00796.pdf)]
<br/><img src="./images/JAKET.jpg" width="700"  alt="model structure"/><br/>

9. **Do Fine-tuned Commonsense Language Models Really Generalize?** *arxiv2020* [[paper](https://arxiv.org/pdf/2011.09159.pdf)]

10. **EFFECTIVE FAQ RETRIEVAL AND QUESTION MATCHING WITH UNSUPERVISEDKNOWLEDGE INJECTION**  *arxiv2020* [[paper](https://arxiv.org/ftp/arxiv/papers/2010/2010.14049.pdf) / [code](https://github.com/DTDwind/TaipeiQA)]

11. **Enriching BERT with Knowledge Graph Embeddings for DocumentClassification** *arxiv2019*
<br/><img src="./images/BERT-Metadata-Author.jpg.jpg" width="700"  alt="model structure"/><br/>

### `Conditional Model`

1. **Conditional BERT Contextual Augmentation** *arxiv2018* [[paper](https://arxiv.org/pdf/1812.06705.pdf)]
<br/><img src="./images/Conditional BERT.jpg" width="700"  alt="model structure"/><br/>

2. **BERT with History Answer Embedding for ConversationalQuestion Answering** *SIGIR2019* [[paper](https://arxiv.org/pdf/1905.05412.pdf)]
<br/><img src="./images/HAE.jpg" width="700"  alt="model structure"/><br/>

### `Theoretical analysis`

1. **Why Do Masked Neural Language Models Still Need Common Sense Knowledge?** *arxiv2019* [[paper](https://arxiv.org/abs/1911.03024)]

2. **How Can We Know What Language Models Know?** *TACL2020* [[paper](https://www.aclweb.org/anthology/2020.tacl-1.28.pdf)]

3. **** **

### `Weak supervisory signal or Marker`

1. **Matching the Blanks: Distributional Similarity for Relation Learning** *ACL2019* [[paper](https://arxiv.org/pdf/1906.03158v1.pdf)]
 <br/><img src="./images/Entity Markers.jpg" width="700"  alt="model structure"/><br/>

2. **MarkedBERT: Integrating Traditional IR Cues in Pre-trainedLanguage Models for Passage Retrieval** *SIGIR2020* [[paper](https://arxiv.org/pdf/1906.03158v1.pdf) / [code](https://github.com/BOUALILILila/markers_bert)]
 <br/><img src="./images/MarkedBERT.jpg" width="400"  alt="model structure"/><br/>
 
3. **GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge** *ENMLP2019* [[paper](https://arxiv.org/pdf/1908.07245.pdf) / [code](https://github.com/HSLCY/GlossBERT)]
 <br/><img src="./images/GlossBERT.jpg" width="500"  alt="model structure"/><br/>
 
4.**Enriching Pre-trained Language Model with Entity Information for Relation Classification** *arxiv2019* [[paper](https://arxiv.org/pdf/1905.08284.pdf)]
 <br/><img src="./images/R-BERT.jpg" width="500"  alt="model structure"/><br/>
 
5. **A Frustratingly Easy Approach for Joint Entity and Relation Extraction** *arxiv2020* [[paper](https://arxiv.org/pdf/2010.12812.pdf)]
 <br/><img src="./images/FAE.jpg" width="700"  alt="model structure"/><br/>
 
 ### `Code Learning`
 1. **Improving Question Answering over Incomplete KBs with Knowledge-Aware Reader** *ACL2019* [[paper](https://arxiv.org/abs/1905.07098) / [code](https://github.com/xwhan/Knowledge-Aware-Reader)]
 <br/><img src="./images/Knowledge-Aware-Reader.jpg" width="700"  alt="model structure"/><br/>
 2. **Knowledge Enhanced Contextual Word Representations(KnowBERT)** *EMNLP2019* [[paper](https://arxiv.org/abs/1909.04164) / [code](https://github.com/allenai/kb)]
  <br/><img src="./images/KnowBERT.jpg" width="600"  alt="model structure"/><br/>
  
 ### `Knowledge or Entity Linker`
 1. [self-supervised entity linker #19](https://github.com/allenai/kb/issues/19)
 2. **Scalable Zero-shot Entity Linking with Dense Entity Retrieval** *EMNLP2019* [[paper](https://arxiv.org/abs/1909.04164) / [code](https://github.com/facebookresearch/BLINK/tree/master/blink)]
  <br/><img src="./images/BLINK.jpg" width="600"  alt="model structure"/><br/>
 3. **Evaluating the Impact of Knowledge Graph Context on Entity Disambiguation Models** *CIKM2020* [[paper](https://arxiv.org/pdf/2008.05190.pdf) / [code](https://github.com/mulangonando/Impact-of-KG-Context-on-ED)]
   <br/><img src="./images/KG Context.jpg" width="800"  alt="model structure"/><br/>
   
 ### `Masked Attention`
 1. **Table Fact Verification with Structure-Aware Transformer** *EMNLP2020* [[paper](https://www.aclweb.org/anthology/2020.emnlp-main.126.pdf) / [code](https://github.com/zhhongzhi/sat)]
  <br/><img src="./images/SAT01.jpg" width="300"  alt="model structure"/><br/>
  <br/><img src="./images/SAT02.jpg" width="600"  alt="model structure"/><br/>
 

## [Retieval-Papers](#content) 
1. **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** *EMNLP 2019* [[paper](https://arxiv.org/pdf/1911.03814.pdf) / [code](https://github.com/UKPLab/sentence-transformers)]
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
<br/><img src="./images/ORQA.jpg" width="800"  alt="model structure"/><br/>

7. **Learning to Learn from Weak Supervision by Full Supervision** *ICML 2018* [[paper](http://metalearning.ml/2017/papers/metalearn17_dehghani.pdf)]
<br/><img src="./images/L2LWS.jpg" width="900"  alt="model structure"/><br/>
8. **Selective Weak Supervision for Neural Information Retrieval** *WWW 2020* [[paper](https://arxiv.org/abs/2001.10382)]

9. **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT** *SIGIR 2020* [[paper](https://arxiv.org/pdf/2004.12832.pdf)]
<br/><img src="./images/ColBERT.jpg" width="900"  alt="model structure"/><br/>

10. **Poly-encoders:architectures and pre-trainingstrategies for fast and accurate multi-sentence scoring** *ICLR 2020* [[paper](https://arxiv.org/pdf/1906.00300.pdf)]
<br/><img src="./images/Poly-encoders01.jpg" width="600"  alt="model structure"/><br/>
<br/><img src="./images/Poly-encoders02.jpg" width="600"  alt="model structure"/><br/>

11. **Learning Robust Models for e-Commerce Product Search** *ACL2020*  [[paper](https://arxiv.org/pdf/2005.03624.pdf)]
<br/><img src="./images/QUARTS.jpg" width="600"  alt="model structure"/><br/>

12. **Word Embeddings for Unsupervised Named Entity Linking** *KSEM2019* [[paper]()]

## [Related papers](#content)
### `Model Fine-tune`
1. **Fine-Tuning Pretrained Language Models:Weight Initializations, Data Orders, and Early Stopping** *arxiv 2020* [[paper](https://arxiv.org/abs/2002.06305)]

2. **Show Your Work: Improved Reporting of Experimental Results** *EMNLP 2019* [[paper](https://arxiv.org/pdf/1909.03004.pdf) / [code](https://github.com/allenai/allentune)]

3. **Showing Your Work Doesn’t Always Work** *ACL 2020* [[paper](https://arxiv.org/pdf/2004.13705.pdf) / [code](https://github.com/castorini/meanmax)]

### `Multi-task Training`
1. **How to Fine-Tune BERT for Text Classification?** *CCL 2019* [[paper](https://arxiv.org/abs/1905.05583) / [code](https://github.com/xuyige/BERT4doc-Classification) / [post](https://zhuanlan.zhihu.com/p/109143667?from_voters_page=true)]

       All the tasks share the BERT layers and the em-bedding layer.  The only layer that does not shareis the final classification layer, which means thateach task has a private classifier layer.
       
### `Curriculum learning`
1. **A Comprehensive Survey on Curriculum Learning** *arxiv2020* [[paper](http://xxx.itp.ac.cn/abs/2010.13166)]
<br/><img src="./images/cl-survey.jpg" width="600"  alt="model structure"/><br/>

2. **Dynamic Curriculum Learning for Imbalanced Data Classification** *ICCV2020* [[paper](https://arxiv.org/pdf/1901.06783.pdf)]

3. **Curriculum Learning by Dynamic Instance Hardness** *NISP2020* [[paper](https://github.com/tianyizhou/DIHCL/blob/main/paper/dihcl_neurips2020_main.pdf)/ [code](https://github.com/tianyizhou/DIHCL)]
<br/><img src="./images/DIHCL.jpg" width="600"  alt="model structure"/><br/>

4. **On The Power of Curriculum Learning in Training Deep Networks** *ICML 2019* [[paper](https://arxiv.org/abs/1904.03626) / [code](https://github.com/GuyHacohen/curriculum_learning)]
<br/><img src="./images/power-of-cl.jpg" width="600"  alt="model structure"/><br/>

5.**Curriculum Learning schedule 总结** [[url](https://www.dazhuanlan.com/2019/12/12/5df15a966a146/)]

### `Contrastive Learning`
1. **Cross-Domain Sentiment Classification withIn-Domain Contrastive Learning** *NIPS2020* [[paper](https://arxiv.org/pdf/2012.02943.pdf]
<br/><img src="./images/Contrastive Learning.jpg" width="600"  alt="model structure"/><br/>


### `Joint Model`
1. [**信息抽取——实体关系联合抽取**](https://www.cnblogs.com/sandwichnlp/p/12049829.html)

## [Competitions](#competitions)

### `Entity Linking`
1. **CCKS2020 任务二：面向中文短文本的实体链指**
[官方链接](http://sigkg.cn/ccks2020/?page_id=516)|
[任务书](http://sigkg.cn/ccks2020/wp-content/uploads/2020/05/2-CCKS2020%E6%8A%80%E6%9C%AF%E8%AF%84%E6%B5%8B-%E9%9D%A2%E5%90%91%E4%B8%AD%E6%96%87%E7%9F%AD%E6%96%87%E6%9C%AC%E7%9A%84%E5%AE%9E%E4%BD%93%E9%93%BE%E6%8C%87.docx)|
[biendata](https://www.biendata.xyz/competition/ccks_2020_el/)|
[Baseline1](https://github.com/PaddlePaddle/Research/tree/master/KG/DuEL_Baseline)|
[Baseline2](https://github.com/nianxw/ccks2020_pytorch_baseline)|
[Baseline3](https://github.com/RegiusQuant/CCKS2020-Entity-Linking)|
[小米知识图谱团队斩获CCKS 2020实体链指比赛冠军](https://mp.weixin.qq.com/s/4s9j-u2607Uo9y3psPeFBg)[vidio](https://hub.baai.ac.cn/view/4130)
[代码参考1](https://github.com/RegiusQuant/CCKS2020-Entity-Linking)

2. **CCKS2019**
[官方链接](http://www.ccks2019.cn/?page_id=62)|
[任务书](https://conference.bj.bcebos.com/ccks2019/eval/CCKS2019-eval-task2.docx)|
[biendata](https://github.com/panchunguang/ccks_baidu_entity_link)|
[CCKS&百度 2019中文短文本的实体链指 第一名解决方案](https://github.com/sunyilgdx/ccks_baidu_entity_link)

3. **百度BROAD Entity Linking**
[官方链接](http://ai.baidu.com/broad/introduction)

4. **CCKS 2020: 面向试验鉴定的命名实体识别任务**
[biendata](https://www.biendata.xyz/competition/ccks_2020_8/)|
[Baseline1](https://github.com/jeffery0628/ccks2020-task8)|
[Baseline2](https://github.com/AI-confused/CCKS2020_Military_NER_Baseline)|

## [会议](#conference)
1. [**CCKS 2020**](http://sigkg.cn/ccks2020/)

[Poster/Demo](https://hub.baai.ac.cn/view/4013)|
[CCKS《前沿技术讲习班》 第22期：「知识图谱专题」资料全公开！](https://hub.baai.ac.cn/view/3931)

## [数据集](#dataset)
1. [MedNLI](https://github.com/jgc128/mednli)
2. [CBLUE](https://github.com/CBLUEbenchmark/CBLUE)

## [数据增强](#data-augment)
1. [给你的数据加上杠杆：文本增强技术的研究进展及应用实践](https://zhuanlan.zhihu.com/p/111882970)
2. [自然语言处理中有哪些常用的数据增强的方式呢？](https://www.zhihu.com/question/305256736)

## [Others](#content)
1. [XLink](https://github.com/solitaryzero/XLink)
2. [OpenMatch](https://github.com/thunlp/OpenMatch)
3. [NeuIRPapers](https://github.com/thunlp/NeuIRPapers)
4. [碎碎念：Transformer的细枝末节](https://zhuanlan.zhihu.com/p/60821628)
5. [BERT模型可以使用无监督的方法做文本相似度任务吗？](https://www.zhihu.com/question/354129879)
6. [Different deterministic behavior between CPU and CUDA for orthogonal initialization](https://github.com/pytorch/pytorch/issues/19013)
7. [知识图谱 | 实体链接](https://zhuanlan.zhihu.com/p/81073607)
8. [论文笔记 | 实体链接：问题、技术和解决方案](https://zhuanlan.zhihu.com/p/82302101)
9. [检索式多轮问答系统模型总结](https://zhuanlan.zhihu.com/p/46366940)
10. [BERT在小米NLP业务中的实战探索](https://blog.csdn.net/weixin_42137700/article/details/105817884)
<br/><img src="./images/xiaomi.jpg" width="300"  alt="model structure"/><br/>
11. [融合知识的检索模型汇总（持续更新）](https://zhuanlan.zhihu.com/p/110371338)
12. [NLP数据增强方法总结：EDA、BT、MixMatch、UDA](https://blog.csdn.net/xixiaoyaoww/article/details/104688002)
13. [BERT meet Knowledge Graph：预训练模型与知识图谱相结合的研究进展](https://zhuanlan.zhihu.com/p/270009212)
