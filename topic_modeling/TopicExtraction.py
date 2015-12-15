from gensim import models
from gensim import corpora
__author__ = 'Paolo'

def main():
	docs = ['aaabacdb abababdb addbaedb daecabdb badbccdb',
			'aeaaacdb abebabdb acdbaedc dbecadda addbbccb',
			'aeaaacdb abebabdb acdbaedc dbecadda addbbccb']
	texts = [[i for i in doc.lower().split()] for doc in docs]
	print texts
	dictionary = corpora.Dictionary(texts)
	dictionary.save('data_topic_modeling/questions.dict');
	# corpus = corpora.TextCorpus(docs)
	corpus = [dictionary.doc2bow(text) for text in texts]
	corpora.MmCorpus.serialize('data_topic_modeling/documents.mm', corpus)
	hdp = models.HdpModel(corpus, dictionary)
	print hdp.show_topics(topics=10,topn=5)

	topicDocs= hdp[corpus]
	for x in topicDocs:
		print x

if __name__ == "__main__":
	main()