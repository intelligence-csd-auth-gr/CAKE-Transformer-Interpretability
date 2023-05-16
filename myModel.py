from transformers import Trainer, TrainingArguments, utils
import numpy as np
from torch.utils.data import Dataset as TDataset
from torch import tensor
from tensorflow.keras.layers import Softmax
from tensorflow.keras.activations import tanh
import torch
import math

class MyDataset(TDataset):
	def __init__(self, encodings, labels, tokenizer):
		self.encodings = tokenizer(list(encodings), truncation=True, padding=True)
		self.labels = labels

	def __getitem__(self, idx):
		item = {key: tensor(val[idx]) for key, val in self.encodings.items()}
		item['labels'] = tensor(self.labels[idx])
		return item

	def __len__(self):
		return len(self.labels)

class MyModel:
	def __init__(self, path, dataset_name, model_name, task, labels, cased, attention=True):
		self.path = path
		self.dataset_name = dataset_name
		self.model_name = model_name
		self.task = task
		self.labels = labels
		self.cased = cased
		self.attention = attention
		self._save_state = {}
		self.__load_model__()
		self.__get_additional_info_from_trainer__()
	
	def __load_model__(self):
		if self.model_name.lower() == 'bert':
			from transformers import BertTokenizerFast
			if self.task.lower() == 'single_label':
				from transformers import BertForSequenceClassification as transformer_model
			else:
				from ourTransformer import BertForMultilabelSequenceClassification as transformer_model
			if self.cased:
				self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
			else:
				self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
		elif self.model_name.lower() == 'distilbert':
			from transformers import DistilBertTokenizerFast
			if self.task.lower() == 'single_label':
				from transformers import DistilBertForSequenceClassification as transformer_model
			else:
				from ourTransformer import DistilBertForMultilabelSequenceClassification as transformer_model
			if self.cased:
				self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
			else:
				self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
		if self.task.lower() == 'single_label':
			model = transformer_model.from_pretrained(self.path+self.dataset_name, output_attentions=self.attention,
															output_hidden_states=self.attention)
		else:
			model = transformer_model.from_pretrained(self.path+self.dataset_name, num_labels = self.labels ,output_attentions=self.attention,
															output_hidden_states=self.attention)
		#utils.logging.disable_progress_bar() 
		training_arguments = TrainingArguments(
			evaluation_strategy='epoch',     # evaluation frequency
			save_strategy='epoch',           # model checkpoint frequency
			logging_strategy='epoch',        # logging frequency
			log_level='critical',             # logging level
			output_dir='./results',          # output directory
			num_train_epochs=1,              # total number of training epochs
			per_device_train_batch_size=8,  # batch size per device during training
			per_device_eval_batch_size=8,   # batch size for evaluation
			warmup_steps=200,                # number of warmup steps for learning rate scheduler
			weight_decay=0.01,               # strength of weight decay
			logging_dir='./logs'             # directory for storing logs
		)
		self.trainer = Trainer(model=model, args=training_arguments)

	def __get_additional_info_from_trainer__(self):
		self.label_weights = np.array(self.trainer.model.classifier.weight.tolist())
		self.key_list = []
		self.query_list = []
		if self.model_name.lower() == 'bert':
			self.layers = 12
			self.heads = 12
			self.embedding_size = 768
			self.ehe = 64 #768/12
			for i in range(self.layers):
				self.key_list.append(self.trainer.model.base_model.encoder.layer[i].attention.self.key.weight.cpu().detach().numpy())
				self.query_list.append(self.trainer.model.base_model.encoder.layer[i].attention.self.query.weight.cpu().detach().numpy())
		else:
			self.layers = 6
			self.heads = 12
			self.embedding_size = 768
			self.ehe = 64 #768/12
			for i in range(self.layers):
				self.key_list.append(self.trainer.model.distilbert.transformer.layer[i].attention.k_lin.weight.cpu().detach().numpy())
				self.query_list.append(self.trainer.model.distilbert.transformer.layer[i].attention.q_lin.weight.cpu().detach().numpy())
		 
	def my_predict(self, instance):
		instance = [instance,instance]
		if self.task == 'single_label':
			instance_labels = [0, 0]
		else:
			instance_labels = [[0]* self.labels,[0]*self.labels]
		instance_dataset = MyDataset(instance, instance_labels, self.tokenizer)
		outputs = self.trainer.predict(instance_dataset)
		if self.attention:
			predictions = outputs.predictions[0]
			hidden_states = np.array(list(outputs.predictions[1]))
			attention_matrix = np.array(list(outputs.predictions[2]))
			#get_additional_info_from_trainer(model_name)
			return predictions[0], attention_matrix[:,0,:,:], hidden_states[:,0,:,:]
		else:
			predictions = outputs.predictions[0]
			return predictions, None, None

	def my_explainFAB(self,instance, config):
		lws_io = config[0] #True: Before multiplying with Q,K, False: After
		every_token = config[1] #True: multiply with CLS, False: multiply with all hidden states
		heads = config[2] #[Mean, Sum] + number of head
		extra_cls = config[3] #Add, Multi, ElementMulti, ''
		activation = config[4] #Softmax, ''
		k_layers = config[5] #Mean, Sum, Multi, number of layer
		matrix = config[6] #From, To, MeanRows, MeanColumns, MaxRows, MaxColumns
		attention_matrix = config[7]
		hidden_states = config[8]
		prediction = config[9]
		
		attention_matrix_dict = {}
		for label in range(self.labels):
			our_new_faboulous_attention = []
			if tuple([label,lws_io,every_token]) not in self._save_state:
				for la in range(self.layers):
					our_new_layer = []
					if self.model_name.lower() == 'bert':
						bob = self.trainer.model.base_model.encoder.layer[la].attention
					elif self.model_name.lower() == 'distilbert':
						bob = self.trainer.model.base_model.transformer.layer[la].attention

					has = hidden_states[la].copy()

					if lws_io:
						if not every_token:
							has[0] = has[0]*self.label_weights[label]# kk = abs(hsf.max()-hsf.min()) try with this instead of 100
						else:
							for ll in range(len(hidden_states[la])):
								has[ll] = has[ll]*self.label_weights[label]                             
								
					if self.model_name.lower() == 'bert':
						aaa = bob.self.key(torch.tensor(has).to('cuda'))
						bbb = bob.self.query(torch.tensor(has).to('cuda'))
					elif self.model_name.lower() == 'distilbert':
						aaa = bob.k_lin(torch.tensor(has).to('cuda'))
						bbb = bob.q_lin(torch.tensor(has).to('cuda'))
						
					if not lws_io:
						if not every_token:
							aaa[0] = torch.tensor(self.label_weights[label]).to('cuda') * aaa[0] 
							bbb[0] = torch.tensor(self.label_weights[label]).to('cuda') * bbb[0]
						else:
							for ll in range(len(hidden_states[la])):
								aaa[ll] = torch.tensor(self.label_weights[label]).to('cuda') * aaa[ll]
								bbb[ll] = torch.tensor(self.label_weights[label]).to('cuda') * bbb[ll]

					for he in range(self.heads):
						if self.model_name.lower() == 'distilbert':
							bbb = bbb / math.sqrt(64)
						attention_scores = torch.matmul(bbb[:,he*self.ehe:(he+1)*self.ehe], aaa[:,he*self.ehe:(he+1)*self.ehe].transpose(-1, -2))
						if self.model_name.lower() == 'lbert':
							attention_scores = attention_scores / math.sqrt(self.ehe)
						attention_scores = attention_scores.cpu().detach().numpy()
						if 'softmax' == activation:
							attention_scores = (Softmax()(attention_scores))        
						our_new_layer.append(attention_scores)
					our_new_faboulous_attention.append(our_new_layer)
				self._save_state[tuple([label,lws_io,every_token])] = our_new_faboulous_attention
			else:
				our_new_faboulous_attention = self._save_state[tuple([label,lws_io,every_token])]   

			attention = np.array(our_new_faboulous_attention)

			if heads == 'Mean':
				attention = attention.mean(axis=1)
			elif heads == 'Sum':
				attention = attention.sum(axis=1)
			elif type(heads) == type(1):
				attention = attention[:,heads,:,:]

			if k_layers == 'Mean':
				attention = attention.mean(axis=0)
			elif k_layers == 'Sum':
				attention = attention.sum(axis=0)
			elif k_layers == 'Multi':
				joint_attention = attention[0]
				for i in range(1, len(attention)):
					joint_attention = np.matmul(attention[i],joint_attention)
				attention = joint_attention
			elif type(k_layers) == type(1):
				attention = attention[k_layers]

			if matrix == 'From':
				attention = attention[0]
			elif matrix == 'To':
				attention = attention[:,0]
			elif matrix == 'MeanColumns':        
				attention = attention.mean(axis=0)
			elif matrix == 'MeanRows':
				attention = attention.mean(axis=1)
			elif matrix == 'MaxColumns':
				attention = attention.max(axis=0)
			elif matrix == 'MaxRows':
				attention = attention.max(axis=1)

			attention_matrix = attention                
			attention_matrix_dict[label] = attention_matrix
		return attention_matrix_dict


	
	def my_explain(self,instance, config):
		lws_io = config[0] #True: Before multiplying with Q,K, False: After
		every_token = config[1] #True: multiply with CLS, False: multiply with all hidden states
		heads = config[2] #[Mean, Sum] + number of head
		extra_cls = config[3] #Add, Multi, ElementMulti, ''
		activation = config[4] #Softmax, ''
		k_layers = config[5] #Mean, Sum, Multi, number of layer
		extract = config[6] #From, To, MeanRows, MeanColumns, MaxRows, MaxColumns
		attention_matrix = config[7]
		hidden_states = config[8]
		prediction = config[9]

		if attention_matrix is None or hidden_states is None or prediction is None:
			_, attention_matrix, hidden_states = self.my_predict(instance)
		
		if heads == 'Mean':
				cls = attention_matrix.copy().mean(axis=1)# -> average heads
		elif heads == 'Sum':
				cls = attention_matrix.copy().sum(axis=1)# -> sum heads
		elif type(heads) == type(1):
				cls = attention_matrix[:,heads,:,:].copy()# -> specific head

		attention_matrix_dict = {}
		for label in range(self.labels):
			final_interpretation = []
			for i in range(self.layers):
				t_hidden_states = hidden_states.copy()

				if lws_io:
					if not every_token:
						t_hidden_states[i, 0] = self.label_weights[label] * hidden_states[i, 0].copy()  # kk = abs(hsf.max()-hsf.min()) try with this instead of 100
					else:
						for ll in range(len(t_hidden_states[i])):
							t_hidden_states[i, ll] = self.label_weights[label] * t_hidden_states[i, ll] 

				temp_hsf = np.matmul(t_hidden_states[i].copy(), self.query_list[i])
				temp_hsfT = np.matmul(t_hidden_states[i].copy(), self.key_list[i])

				if not lws_io:
					if not every_token:
						temp_hsf[0] = self.label_weights[label] * temp_hsf[0] 
						temp_hsfT[0] = self.label_weights[label] * temp_hsfT[0]
					else:
						for ll in range(len(temp_hsf)):
							temp_hsf[ll] = self.label_weights[label] * temp_hsf[ll]
							temp_hsfT[ll] = self.label_weights[label] * temp_hsfT[ll] 

				if 'softmax' == activation:
					temp = (Softmax()((np.dot(temp_hsf, temp_hsfT.T)) / np.sqrt(self.embedding_size)))
				else: 
					temp = ((np.dot(temp_hsf, temp_hsfT.T)) / np.sqrt(self.embedding_size)) 
				
				if extra_cls == 'Add':
						temp = temp + cls[i]
				elif extra_cls == 'Multi':
						temp = np.matmul(temp,cls[i])
				elif extra_cls == 'ElementMulti':
						temp = np.multiply(temp,cls[i])
				else:
					temp = temp
					
				final_interpretation.append(temp)
			final_interpretation = np.array(final_interpretation)

			if k_layers == 'Mean':
				final_interpretation = final_interpretation.mean(axis=0)
			elif k_layers == 'Sum':
				final_interpretation = final_interpretation.sum(axis=0)
			elif k_layers == 'Multi':
				joint_attention = final_interpretation[0]
				for i in range(1, len(final_interpretation)):
					joint_attention = np.matmul(final_interpretation[i],joint_attention)
				final_interpretation = joint_attention
			else:
				final_interpretation = final_interpretation[k_layers]

			if extract == 'From':
				final_interpretation = final_interpretation[0]
			elif extract == 'To':
				final_interpretation = final_interpretation[:,0]
			elif extract == 'MeanColumns':        
				final_interpretation = final_interpretation.mean(axis=0)
			elif extract == 'MeanRows':
				final_interpretation = final_interpretation.mean(axis=1)
			elif extract == 'MaxColumns':
				final_interpretation = final_interpretation.max(axis=0)
			elif extract == 'MaxRows':
				final_interpretation = final_interpretation.max(axis=1)

			attention_matrix = final_interpretation             
			attention_matrix_dict[label] = attention_matrix
		return attention_matrix_dict
