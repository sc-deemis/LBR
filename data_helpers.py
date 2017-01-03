import tensorflow as tf
import pandas as pd
import numpy as np
import config
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import word_tokenize
import sys

INPUT_FILE = config.INPUT_FILE
OUTPUT_FILE = config.OUTPUT_FILE
NER_MODEL_PATH = config.NER_MODEL_PATH
NER_JAR_PATH = config.NER_JAR_PATH


reload(sys)  
sys.setdefaultencoding('utf8')

def load_normalized_data(INPUT_FILE):
	data = pd.read_csv(INPUT_FILE)
        fee_expense = data.loc[data['Category'].isin(['Fee','Expense'])]
	adjustment = data.loc[data['Category'].isin(['Adjustment'])]
	classifier = fee_expense.merge(adjustment,left_on='InvoiceLineItemId',right_on='ParentInvoiceLineItemId',how='outer')
	classifier = classifier.fillna('missing')
	classifier = classifier[classifier.InvoiceLineItemId_x != 'missing']
	classifier = classifier[['InvoiceId_x', 'VendorId_x', 'InvoiceStatus_x',
       'InvoiceLineItemId_x', 'NetTotalAmount_x', 'CodeDescription_x',
       'Category_x', 'AdjNarrative_x', 'Units_x', 'Rate_x',
       'ApprovedAmount_x', 'NetTotalAmount_y',
       'ParentInvoiceLineItemId_y', 'CodeDescription_y',
       'Category_y', 'AdjNarrative_y',
       'Hours_y', 'BilledFees_y', 'BilledExpenses_y',
       'ApprovedAmount_y']]
	classifier.columns = ['InvoiceId', 'VendorId', 'InvoiceStatus',
       'InvoiceLineItemId', 'NetTotalAmount', 'BeforeCodeDescription',
       'BeforeCategory', 'InputNarrative', 'Units', 'Rate',
       'BeforeApprovedAmount', 'AfterNetTotalAmount',
       'ParentInvoiceLineItemId', 'AfterCodeDescription',
       'AfterCategory', 'AdjNarrative',
       'Hours', 'AfterBilledFees', 'AfterBilledExpenses',
       'AfterApprovedAmount']

	classifier['Label'] = np.where(classifier['AfterCategory']=='Adjustment', 1,0)
	return classifier.to_csv(OUTPUT_FILE,sep=",")

def load_train_test():
	classifier = load_data('/home/admin8899/LBR/data/SubChubbFile.csv')
	data = classifier.loc[classifier['BeforeCodeDescription'].isin(['Travel'])]
	data['InvoiceStatus'] = pd.factorize(data.InvoiceStatus)[0] + 1
	data = data[['InvoiceStatus','NetTotalAmount','BeforeApprovedAmount','InputNarrative','Label']]
	train_data = data[:-1500]
	test_data = data[-1500:]
	test_data.to_csv("./travel_test.csv",sep=',')
	x_num = train_data[['InvoiceStatus','NetTotalAmount','BeforeApprovedAmount']]
	x_num = x_num.as_matrix()
	x_text = train_data[['InputNarrative']]
	x_text = ner(x_text)
	print(x_text)
	y = train_data[['Label']]
	y = y['Label'].tolist()
	#y = [[0, 1] for i in y if i==1 else [1,0]]
	yy=[]
	for i in y:
		if i ==1:
			yy.append([0,1])
		else:
			yy.append([1,0])
	y=np.array(yy)
	return [x_text,x_num,y]

def load_normalized_annotated_data():
	data = pd.read_csv(OUTPUT_FILE)
	data['InvoiceStatus'] = pd.factorize(data.InvoiceStatus)[0] + 1
	x_text = data[['InputNarrative']]
	x_text = x_text['InputNarrative'].tolist()
	data['AnnotatedNarrative'] = pd.DataFrame(ner(x_text))
	return data.to_csv('./normalized_annotated_data.csv',sep=",")


def ner(x_text):
        st = StanfordNERTagger(NER_MODEL_PATH,NER_JAR_PATH,encoding='utf-8')
	annotated_text=[]
        for ind,i in enumerate(x_text):
                tokenized_text = word_tokenize(i)
                classified_text = dict(st.tag(tokenized_text))
                classified_text = {i:classified_text[i] for i in classified_text if classified_text[i] in ('PERSON','LOCATION','ORGANIZATION')}
                my_lists = [classified_text.get(x,x) for x in  tokenized_text]
		print(ind)
		print (" ".join(my_lists))
                annotated_text.append(" ".join(my_lists))
	return annotated_text


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

#load_normalized_data('/home/admin8899/LBR/data/SubChubbFile.csv')
load_normalized_annotated_data()
