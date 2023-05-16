import lime
import torch
from scipy.special import softmax
from lime.lime_text import LimeTextExplainer
from transformers_interpret import SequenceClassificationExplainer
import numpy as np
from myModel import MyDataset
import tensorflow as tf

class MyExplainer:
	"""docstring for ClassName"""

	def __init__(self, label_names, model, sentence_level=False, split_token='.', layers=12, heads=12, cake = None):
		self.layers = layers
		self.heads=heads
		self.label_names = label_names
		self.tokenizer = model.tokenizer
		self.sentence_level = sentence_level
		self.split_token = split_token
		self.model = model
		self.max_sequence_len = self.tokenizer.max_len_single_sentence
		self.config = None
		self.lime_explainer = LimeTextExplainer(class_names=label_names, split_expression='\s+', bow=False)
		self.ig_explainer = SequenceClassificationExplainer(self.model.trainer.model, self.tokenizer,
															custom_labels=self.label_names)
		self.save_states = {}
		self.neighbours = 2000
		self.cake_conf = [1, 1, None, 0, False]
		self.cake = cake
		self.found_ratios = {}
		
	def fix_instance(self, instance):
		new_sentence = ''
		temp_split = instance.split()
		for i in range(0,len(temp_split)):
			if "##" not in temp_split[i]:
				new_sentence = new_sentence + ' ' + temp_split[i]
			else:
				new_sentence = new_sentence + temp_split[i][2:]
		return new_sentence[1:]

	def convert_to_sentence(self, tokens, interpretations):
		if self.split_token != '.':
			abstract = ' '.join(tokens)+' '
			abstract = abstract.replace(' . ',' ‡ ')
			abstract = abstract.replace('.',' ‡ ')
			tokens = abstract.split()
		label_interpretations = []
		label_sentences = []
		for label in range(len(self.label_names)):
			sentences = []
			sentences_weights = []
			sentence = []
			sentence_weight = []
			flagoto= False
			for weight,token in zip(interpretations[label],tokens[1:-1]):
				if token == '‡' and flagoto:
					sentences.append(' ')
					sentences_weights.append(0.0) #mean/max/sum?whatever helps us...
					sentence = []
					sentence_weight = []
				else:
					if self.split_token != '.' and token == '‡':
						sentence.append('.')
						flagoto = True
					else:
						sentence.append(token)
						flagoto = False
					sentence_weight.append(weight)
					if token == self.split_token:
						sentences.append(self.fix_instance(' '.join(sentence)))
						sentences_weights.append(np.array(sentence_weight).mean()) #mean/max/sum?whatever helps us...
						sentence = []
						sentence_weight = []
				label_sentences.append(sentences)
			label_interpretations.append(sentences_weights)
		return [label_sentences, label_interpretations]

	def lime(self, instance, prediction, tokens, mask, attention, hidden_states):
		def predictor(texts):
			all_probabilities = []
			splits = np.array_split(texts, 200)
			for split in splits:
				if self.model.task == 'single_label':
					split_labels = [0] * len(split)
				else:
					tt = []
					for s in split:
						split_labels = [0] * len(self.label_names)
						tt.append(split_labels)
					split_labels = np.array(tt)
				dataset = MyDataset(split, split_labels, self.tokenizer)
				logits, _, _ = self.model.trainer.predict(dataset)[0]
				if len(self.label_names) == 2:
					probabilities = softmax(logits, axis=1)
				else:
					a = tf.constant(logits, dtype = tf.float32)
					b = tf.keras.activations.sigmoid(a)
					probabilities = b.numpy()
				all_probabilities.extend(probabilities)
			return np.array(all_probabilities)

		temp_instance = ' '.join(tokens)
		interpretations = []
		for label in range(len(self.label_names)):
			exp = self.lime_explainer.explain_instance(temp_instance, predictor, num_features=self.max_sequence_len,
													   num_samples=self.neighbours, labels=(label,))
			explanation_dict = dict(list(exp.as_map().values())[0])
			scores = []
			for i in range(1, len(tokens) - 1):
				scores.append(explanation_dict[i])
			interpretations.append(scores)
		if self.sentence_level:
			interpretations = self.convert_to_sentence(tokens, interpretations)
		return interpretations

	def ig(self, instance, prediction, tokens, mask, attention, hidden_states):
		interpretations = []
		for label in range(len(self.label_names)):
			explanations = [explanation[1] for explanation in self.ig_explainer(instance, index=label, internal_batch_size=10, n_steps=50)[1:-1]]
			interpretations.append(explanations)
		if self.sentence_level:
			interpretations = self.convert_to_sentence(tokens, interpretations)
		return interpretations

	def cake_explain(self, instance, prediction, tokens, mask, attention, hidden_states):
		key_emb = self.cake_conf[0]
		label_emb = self.cake_conf[1]
		pid = self.cake_conf[2]
		keyphrases = self.cake_conf[3]
		width = self.cake_conf[4]
		negatives = self.cake_conf[5]
		interpretations, found_ratio = self.cake.keyphrase_interpretation(instance, keyphrases, key_emb, label_emb, width, negatives, pid)
		key = ' '.join(str(i) for i in self.cake_conf)
		if key not in self.found_ratios:
			self.found_ratios[key] = [found_ratio]
		else:
			self.found_ratios[key].append(found_ratio)
		if self.sentence_level:
			interpretations = self.convert_to_sentence(tokens, interpretations)
		return interpretations 

	def our_explain(self, instance, prediction, tokens, mask, attention, hidden_states):
		interpretations = []
		for label in range(len(self.label_names)):
			interpretations.append(self.model.my_explain(instance, self.config)[label][1:-1])
		if self.sentence_level:
			interpretations = self.convert_to_sentence(tokens, interpretations)
		return interpretations
	
	def our_explainFAB(self, instance, prediction, tokens, mask, attention, hidden_states):
		interpretations = []
		for label in range(len(self.label_names)):
			interpretations.append(self.model.my_explainFAB(instance, self.config)[label][1:-1])
		if self.sentence_level:
			interpretations = self.convert_to_sentence(tokens, interpretations)
		return interpretations

	def my_attention(self, instance, prediction, tokens, mask, attention_i, hidden_states):
		layers = self.config[0] # Mean, Multi, Sum, First, Last
		heads = self.config[1] # Mean, Sum, First, Last
		matrix = self.config[2] # From, To, MeanColumns, MeanRows, MaxColumns, MaxRows
		selection = self.config[3] #True: select layers per head, False: do not
		
		attention = attention_i.copy()
		if not selection:
			
			if heads == 'Mean':
				if heads not in self.save_states:
					self.save_states[heads] = attention.mean(axis=1)
					attention = self.save_states[heads]
				else:
					attention = self.save_states[heads]
			elif heads == 'Sum':
				if heads not in self.save_states:
					self.save_states[heads] = attention.sum(axis=1)
					attention = self.save_states[heads]
				else:
					attention = self.save_states[heads]
			elif type(heads) == type(1):
				attention = attention[:,heads,:,:]
				
			if layers == 'Mean':
				attention = attention.mean(axis=0)
			elif layers == 'Sum':
				attention = attention.sum(axis=0)
			elif layers == 'Multi':
				joint_attention = attention[0]
				for i in range(1, len(attention)):
					joint_attention = np.matmul(attention[i],joint_attention)
				attention = joint_attention
			elif type(layers) == type(1):
				attention = attention[layers]

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
		else:
			importance_attention_matrices = []
			for i in range(self.layers): #TO VHANGE IN THE FUTURE?
				att_heads = []
				for j in range(self.heads): #TO VHANGE IN THE FUTURE?
					mm = attention_i[i][j][1:-1,1:-1].max()
					if mm > 0.5:
						indi = 0
						indj = 0
						for k in np.argmax(attention_i[i][j][1:-1,1:-1],axis=0):
							if mm in attention_i[i][j][1:-1,1:-1][k]:
								indi = k
								indj = np.argmax(attention_i[i][j][1:-1,1:-1][k])
						if abs(indi-indj) != 0:
							att_heads.append(attention_i[i][j])
				if heads == 'Mean' and len(att_heads) > 0:
						importance_attention_matrices.append(np.array(att_heads).mean(axis=0))
				elif heads == 'Sum' and len(att_heads) > 0:
						importance_attention_matrices.append(np.array(att_heads).sum(axis=0))
			importance_attention_matrices = np.array(importance_attention_matrices)

			if layers == 'Mean':
				attention = importance_attention_matrices.mean(axis=0)
			elif layers == 'Sum':
				attention = importance_attention_matrices.sum(axis=0)
			elif layers == 'Multi':
				attention = importance_attention_matrices[0]
				for i in range(1,len(importance_attention_matrices)):
					attention = np.matmul(attention,importance_attention_matrices[i])
			attention=attention[0]
		interpretations = []
		for label in range(len(self.label_names)):
			interpretations.append(attention[1:-1])
		if self.sentence_level:
			interpretations = self.convert_to_sentence(tokens, interpretations)
		return interpretations