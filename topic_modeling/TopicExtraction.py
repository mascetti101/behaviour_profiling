from gensim import models
from gensim import corpora
from WordData import WordData
import pandas as pd
import numpy as np

__author__ = 'Paolo'

def main():

	newData = pd.read_csv('../xsense_data/global_dataset.txt', sep=';')

	###############################LONG WORD TRY ###############################
	############################### 15 SIGNALS   ###############################
	## Choose feature to represent in words
	## All exclused altitude
	## dataPartOne = newData.ix[:,'Acc_X':'Pitch']
	## dataPartTwo = newData.ix[:, 'Speed_X':'Speed_Z']

	## newDataToWord = pd.concat([dataPartOne,dataPartTwo], axis=1)
	###############################REDUCED WORD TRY ###############################
	############################### 5 SIGNALS       ###############################
	newDataToWord = newData.ix[:,['Acc_X','Acc_Y','Acc_Z','Speed_X','Roll']]

	worder = WordData(newDataToWord)
	words = worder.create_words(worder.dataset)

	colWords = pd.Series(words, name='Word')
	wordDataset = pd.concat([newData,colWords], axis=1)
	wordDataset.to_csv('../xsense_data/word_global_dataset.txt',sep=';')

	docs = worder.create_text_corpus(wordDataset)

	#docs = ['aaabacdb abababdb addbaedb daecabdb badbccdb',
	#		'aeaaacdb abebabdb acdbaedc dbecadda addbbccb',
	#		'aeaaacdb abebabdb acdbaedc dbecadda addbbccb']

	texts = [[i for i in doc.lower().split()] for doc in docs]

	dictionary = corpora.Dictionary(texts)
	dictionary.save('data_topic_modeling/doc_dictionary.dict');
	# corpus = corpora.TextCorpus(docs)
	corpus = [dictionary.doc2bow(text) for text in texts]
	corpora.MmCorpus.serialize('data_topic_modeling/documents.mm', corpus)
	hdp = models.HdpModel(corpus, dictionary, T=50, K=10)
	print hdp.show_topics(topics=20, topn=5)

	topicDocs= hdp[corpus]
	for x in topicDocs:
		print x

	alpha, beta = hdp.hdp_to_lda()
	print alpha
	lda_model = models.LdaModel(id2word=hdp.id2word,
						num_topics=len(alpha),
						alpha=alpha,
						eta=hdp.m_eta)
	lda_model.expElogbeta = np.array(beta, dtype=np.float32)
	print lda_model.show_topic(1)

def main_new_dataset():

	newData = pd.read_csv('../xsense_data/global_dataset_abs_speed_diff_yaw.txt', sep=';')
	newDataToWord = newData.ix[:,['Acc_X','Acc_Y','Speed_X','Speed_Y','Diff_Yaw']]

	worder = WordData(newDataToWord)
	words = worder.create_words(worder.dataset)
	colWords = pd.Series(words, name='Word')
	wordDataset = pd.concat([newData,colWords], axis=1)
	#wordDataset.to_csv('../xsense_data/word_global_dataset.txt',sep=';')

	docs = worder.create_text_corpus(wordDataset)

	texts = [[i for i in doc.lower().split()] for doc in docs]

	dictionary = corpora.Dictionary(texts)
	dictionary.save('data_topic_modeling/new_dataset/doc_dictionary.dict');
	# corpus = corpora.TextCorpus(docs)
	corpus = [dictionary.doc2bow(text) for text in texts]
	corpora.MmCorpus.serialize('data_topic_modeling/new_dataset/documents.mm', corpus)
	hdp = models.HdpModel(corpus, dictionary, T=50, K=10)
	print hdp.show_topics(topics=20, topn=5)

	topicDocs= hdp[corpus]
	for x in topicDocs:
		print x

if __name__ == "__main__":
	main_new_dataset()