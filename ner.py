
import numpy as np
import re, subprocess, os, shutil, math, re, logging, pickle, json, itertools, functools, psutil, time, random
import pandas
import spacy
import tensorflow.keras as keras
import crf
import tensorflow as tf
import torch, cupy 
import h5py 
import annotations


# List of allowed characters (for the character embeddings)
CHARACTERS = [chr(i) for i in range(32, 127)] + [chr(i) for i in range(160,255)] + ['—','―','‘','’','“','”','…','€']

# Output labels
LABELS = ['CARDINAL', "COMPANY", 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 
          'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']


class NERModel(annotations.BaseAnnotator):
    """
    A Keras implementation for sequence labeling, including both traditional word embeddings, 
    character embeddings, and contextual word embeddings from RoBERTa. The neural architecture
    can make use of convolutional or recurrent (e.g. BiLSTM) layers, as well as an optional 
    CRF layer. 
    """
    
    
    def __init__(self, name="neural-ner-model", model_file=None, **kwargs):
        self.params = {
            "vocab_size":20000,                 # size of the (token) vocabulary
            "trainable_word_embeddings":True,   # Whether to tune the word embeddings or keep them fixed
            "word_emb_transform_dim":0,         # Dimension of the dense layer transforming the word embeddings (0 to skip)
            "char_embedding_dim":48,            # Dimension for the character embeddings (0 to skip the layer)
            "normalise_chars":False,            # Whether to normalise the characters in the tokens
            "max_token_length":32,              # Maximum number of characters per token (for the character embeddings)
            "char_lstm_dim":48,                 # Dimension for the character-level biLSTM (0 to skip the layer)

            "use_roberta_embeddings":False,     # Whether to also include roberta embeddings

            "nb_convo_layers":0,                # Number of token-level convolutional layers (0 for no layers)
            "token_kernel_size":5,              # Kernel size for the convolutional layers
            "token_filter_dim":128,             # Filter size for the convolutional layers
            "token_lstm_dim":128,               # Dimension for the token-level biLSTM (0 to skip the layer)
            "dense_dim":0,                      # Size of the dense layer (after convolutions and LSTMs)
            "use_crf":False,                     # Whether to use a CRF as final layer (0 to skip the layer)

            "dropout":0.3,                      # Dropout ratio (on all embeddings)
            "optimiser":"Adam",                 # Optimisation algorithm
            "epoch_length":50000,               # number of training examples per epoch
            "nb_epochs":5,                      # number of epochs
            "lr":0.001,                         # learning rate
            "batch_size":1,                     # Number of documents per batch
            "gpu":0                             # GPU index
        } 
        self.params.update(kwargs)
        self.name = name

        # Setting up the GPU
        if self.params["gpu"] is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.params["gpu"])
       #     gpu =  tf.config.experimental.list_physical_devices('GPU')[self.params["gpu"]]
      #      tf.config.experimental.set_memory_growth(gpu, True)     
        
        # We load the Spacy standard model
        self.nlp = spacy.load("en_core_web_md")

        # If we need to reload an existing model
        if model_file is not None:
            f = h5py.File(model_file, 'r')
            self.name = f.attrs["name"]
            self.indices = json.loads(f.attrs['indices'])
            self.char_indices = json.loads(f.attrs['char_indices'])         
            self.label_indices = json.loads(f.attrs['label_indices'])   
            self.label_indices_inverted = {i:l for l,i in self.label_indices.items()}
            self.params.update(json.loads(f.attrs['params']))
            self.params.update(kwargs)
            self.model = tf.keras.models.load_model(model_file)
            f.close() 
            
        else:
            # Token indices  
            # CONVENTION: 0 is padding value and 1 for indices of unknown tokens
            self.indices = {x.norm_:(x.rank+2) for x in self.nlp.vocab if x.has_vector 
                            and x.rank < (self.params["vocab_size"]-2)}

            # Character indices
            self.char_indices = {c:i+2 for i, c in enumerate(CHARACTERS)}

            # Output label indices
            self.label_indices = {"O":0}
            self.label_indices_inverted = {0:"O"}
            for label in LABELS:
                for biluo in "BILU":
                    index = len(self.label_indices)
                    full_label = "%s-%s"%(biluo, label)
                    self.label_indices[full_label] = index
                    self.label_indices_inverted[index] = full_label
                    
            # Empty model for now
            self.model = None
                 
        if self.params["use_roberta_embeddings"]:
            if self.params["gpu"] is not None:
                is_using_gpu = spacy.prefer_gpu()
                if is_using_gpu:
                    print("Using GPU for RoBERTa...")
                    torch.set_default_tensor_type("torch.cuda.FloatTensor")
            self.roberta = spacy.load("en_trf_robertabase_lg")
 
 
 
    def train(self, train_docs, val_docs):
        """Trains the NER model on Spacy documents"""
        
        print("Training NER model for %i labels:"%len(LABELS), LABELS)
        print("Parameters:", self.params)
                
        # Builds the model
        self.build()
        self.model.summary()
        
        # The model is saved at every epoch
        saver = keras.callbacks.ModelCheckpoint("tmp_model.h5")

        # Prepares validation data
        print("Processing validation data", end="...", flush=True)
        val_batch = _stack_batch(list(self.generator(val_docs)))
        print("done")
        
        # Starts training
        print("start training")
        batch_generator = self.generator(train_docs)
        self.model.fit_generator(batch_generator, validation_data=val_batch,
                                 steps_per_epoch=self.params["epoch_length"],
                                 epochs=self.params["nb_epochs"],callbacks=[saver], 
                                 use_multiprocessing=True, max_queue_size=150) 
        
        return self
        
       
        
    def build(self):
        """Builds the neural architecture (based on the parameters in self.params)"""
        
        inputs = []
        embeddings = []
        word_ids = keras.layers.Input(shape=(None,), dtype='int32', name='word_input')
        inputs.append(word_ids)
        
        # Initialise the embedding layer with the pretrained embeddings from Spacy 
        initial_embeddings = np.zeros((self.params["vocab_size"], self.nlp.vocab.vectors_length))
        for x in self.nlp.vocab:
            if x.norm_ in self.indices:
                initial_embeddings[self.indices[x.norm_],:] = x.vector
              
        # Construct the token-level embeddings
        embedding_layer = keras.layers.Embedding(input_dim=initial_embeddings.shape[0], 
                                                 output_dim=initial_embeddings.shape[1], 
                                                 trainable=self.params["trainable_word_embeddings"],
                                                 embeddings_initializer=keras.initializers.Constant(initial_embeddings),
                                                 mask_zero=False, name='word_embedding')
        word_embeddings = embedding_layer(word_ids)
        if self.params["word_emb_transform_dim"]:
            transform = keras.layers.Dense(self.params["word_emb_transform_dim"], activation="relu",
                                          name ="word_embeddings_transform")
            word_embeddings = transform(word_embeddings)
        embeddings.append(word_embeddings)

        # build character-based embeddings
        if self.params["char_embedding_dim"]:
            char_ids = keras.layers.Input(shape=(None, self.params["max_token_length"]), dtype='int16', name='char_input')
            inputs.append(char_ids)
            char_embedding_layer = keras.layers.Embedding(input_dim=len(CHARACTERS)+1,
                                                          output_dim=self.params["char_embedding_dim"],
                                                          mask_zero=False, name='char_embedding')
            char_embeddings = char_embedding_layer(char_ids) 
            
            # Build a biLSTM at the character level
            if self.params["char_lstm_dim"]:
                char_lstm_layer = keras.layers.LSTM(self.params["char_lstm_dim"])
                char_bilstm_layer = keras.layers.Bidirectional(char_lstm_layer)
                distributed_layer = keras.layers.TimeDistributed(char_bilstm_layer, name="char_lstm")
                char_lstm_embeddings = distributed_layer(char_embeddings)
                embeddings.append(char_lstm_embeddings)
                
            # Otherwise, do a max-pooling on the character embeddings
            else:
                # NOTE: we should perform masking to avoid taking 0 values into account
                char_pooling_layer = keras.layers.GlobalMaxPooling1D()
                distributed_layer = keras.layers.TimeDistributed(char_pooling_layer, name="char_pooling")
                char_pooled_embeddings = distributed_layer(char_embeddings)
                embeddings.append(char_pooled_embeddings)

        # Use context-sensitive word embeddings from roBERTa
        if self.params["use_roberta_embeddings"]:
            roberta_embeddings = keras.layers.Input(shape=(None, 768), dtype='float32', name="roberta_embeddings")
            inputs.append(roberta_embeddings)
            embeddings.append(roberta_embeddings)

        # Concatenate all the embeddings (usual word embeddings + character-level embeddings + roBerta) into one large vector
        if len(embeddings) > 1:
            token_embeddings = keras.layers.Concatenate(axis=-1)(embeddings)
        else:
            token_embeddings = embeddings[0]
            
        # Perform dropout
        dropout = keras.layers.Dropout(self.params["dropout"], name="token_dropout")
        token_embeddings = dropout(token_embeddings)

        # Add convolutional layers
        for i in range(self.params["nb_convo_layers"]):
            convo = keras.layers.Conv1D(self.params["token_filter_dim"], kernel_size=self.params["token_kernel_size"], 
                                        padding='same', activation="relu", name="token_convolution_%i"%(i+1))
            token_embeddings = convo(token_embeddings)

        # Add a biLSTM layer
        if self.params["token_lstm_dim"]:
            token_lstm_layer = keras.layers.LSTM(self.params["token_lstm_dim"], return_sequences=True)
            token_bilstm_layer = keras.layers.Bidirectional(token_lstm_layer, name="token_biLSTM")
            token_embeddings = token_bilstm_layer(token_embeddings)            

        # Add a dense layer after convolutions + biLSTM
        if self.params["dense_dim"]:
            dense_layer = keras.layers.Dense(self.params["dense_dim"], activation="relu", name="token_dense")
            final_token_embeddings = dense_layer(token_embeddings)      
        else:
            final_token_embeddings = token_embeddings
        
        # Create final layer (CRF or softmax layer)
        if self.params["use_crf"]:
            output_layer = crf.CRF(len(self.label_indices), 
                                   learn_mode="marginal", test_mode="marginal", name="crf_output")
            loss = output_layer.loss_function
        else:
            output_layer = keras.layers.Dense(len(self.label_indices), 
                                              name="softmax_output", activation="softmax")
            loss = "categorical_crossentropy"
          
        # Create final model
        output = output_layer(final_token_embeddings)
        self.model = keras.models.Model(inputs=inputs, outputs=output)
        
        optimiser = getattr(keras.optimizers, self.params["optimiser"])(lr=self.params["lr"])
        self.model.compile(loss=loss, optimizer=optimiser, weighted_metrics=["categorical_accuracy"], sample_weight_mode="temporal")

        return self.model
    
    
    def save(self, model_file):
        """Saves the model into a HDF5 file"""
        
        if self.model is None:
            raise RuntimeError("Model is not yet trained!")
        
        # Add some method data to the file
        self.model.save(model_file) 
        f = h5py.File(model_file, 'a')
        f.attrs['name'] = self.name
        f.attrs['indices'] = json.dumps(self.indices)
        f.attrs['char_indices'] = json.dumps(self.char_indices)
        f.attrs['label_indices'] = json.dumps(self.label_indices)
        f.attrs['params'] = json.dumps(self.params)
        f.close()
        print("Model saved to", model_file)  
        
        return self

        
    def _check_outputs(self, predictions):
        """Checks whether the output is consistent"""
        
        prev_bilu_label = "O"
        for i in range(len(predictions)):
            bilu_label = self.label_indices_inverted[predictions[i]]
            if prev_bilu_label[0] in {"L", "U", "O"} and bilu_label[0] in {"I", "L"}:
                print("inconsistent start of NER at pos %i:"%i, bilu_label, "after", prev_bilu_label)
            elif prev_bilu_label[0] in {"B", "I"}:
                if bilu_label[0] not in {"I", "L"} or bilu_label[2:]!=prev_bilu_label[2:]:
                    print("inconsistent continuation of NER at pos %i:"%i, bilu_label, "after", prev_bilu_label)
            prev_bilu_label = bilu_label
                    
                      
    def annotate(self, spacy_doc, replace=False):
        """Run the NER model. If replace is True, the named entities in the document are replaced.
        Otherwise, the method adds the annotations as a new layer in the user-data 
        (with self.name as source name)"""
        
        inputs = self._convert_inputs(spacy_doc)
        probs  =self.model.predict(inputs)[0]
        predictions = probs.argmax(axis=1)
    #    self._check_outputs(predictions)
        predict_probs = probs.max(axis=1)
        
        spans = []
        i = 0
        while i < len(predictions):
            if predictions[i]==0:
                i += 1
                continue
            bilu_label = self.label_indices_inverted[predictions[i]]
            label = bilu_label[2:]
            if bilu_label[0]=="U":
                spans.append((i,i+1, label, predict_probs[i]))
                i += 1
            else:
                start = i
                i += 1
                while i < len(predictions)-1 and self.label_indices_inverted[predictions[i]][0] in {"I", "L"}:
                    i += 1
                spans.append((start,i, label, predict_probs[start:i].mean()))
                
        if replace is True:
            spacy_doc.ents = [spacy.tokens.Span(spacy_doc, start, end, spacy_doc.vocab.strings[label])
                              for start, end, label,_ in spans]
        else:
            annotated = {(start,end):((label, conf),) for start,end,label, conf in spans}
            if "annotations" not in spacy_doc.user_data:
                spacy_doc.user_data["annotations"] = {self.name: annotated}
            else:
                spacy_doc.user_data["annotations"][self.name] = annotated
                
        return spacy_doc
                          
    def generator(self, docs):
        """Generates the input tensors for each document """
               
        # If we use roBERTa embeddings, run the docs through a pipe
        if self.params["use_roberta_embeddings"]:
           
            roberta_wordpiecer = self.roberta.get_pipe("trf_wordpiecer")
            roberta_tok2vec = self.roberta.get_pipe("trf_tok2vec")
            docs = roberta_tok2vec.pipe(roberta_wordpiecer.pipe(docs))
        
        batch = []
        for spacy_doc in docs:

            inputs = self._convert_inputs(spacy_doc)
            outputs = self._convert_outputs(spacy_doc)
            weights = np.ones((1, len(spacy_doc)), dtype=np.float32)
                        
            if self.params["batch_size"]==1:
                yield inputs, outputs, weights
            else:
                batch.append((inputs, outputs, weights))
                if len(batch)>= self.params["batch_size"]:
                    yield _stack_batch(batch)
                    batch.clear()
        if batch:
            yield _stack_batch(batch)
 

    def _convert_inputs(self, spacy_doc):
        # First extract the token-level indices
        token_indices = np.array([[self.indices.get(token.norm_, 1) for token in spacy_doc]], dtype=np.int32)
        inputs = [token_indices]
               
        # Then the character level indices
        if self.params["char_embedding_dim"]:
            char_inputs = np.zeros((1,len(spacy_doc), self.params["max_token_length"]), dtype=np.int16)
            for token in spacy_doc:
                token2 = token.norm_ if self.params["normalise_chars"] else token.text
                token_length = min(self.params["max_token_length"], len(token2))
                char_inputs[0,token.i,:token_length] = [self.char_indices.get(c,1) for i, c in enumerate(token2) if i < token_length]
            inputs.append(char_inputs)
              
        # And the roBERTa indices
        if self.params["use_roberta_embeddings"]:
            roberta_embeddings = cupy.asnumpy(spacy_doc.tensor)
            inputs.append(np.expand_dims(roberta_embeddings,axis=0))
        return inputs
    
    
    def _convert_outputs(self, spacy_doc):
        labels = np.zeros((1,len(spacy_doc), len(self.label_indices)), dtype=np.float32)
        for ent in spacy_doc.ents:
            conf = 1.0 # Right now we don't use probabilistic labels
            if ent.start > len(spacy_doc):
                print("wha??", list(spacy_doc), ent.start, ent.end)
            if ent.start==ent.end-1:
                labels[0,ent.start,self.label_indices["U-%s"%ent.label_]] = conf
            else:
                labels[0,ent.start,self.label_indices["B-%s"%ent.label_]] = conf 
                for i in range(ent.start+1, ent.end-1):
                    labels[0,i,self.label_indices["I-%s"%ent.label_]] = conf 
                labels[0,ent.end-1,self.label_indices["L-%s"%ent.label_]] = conf
        labels[0,:,0] = 1-labels[0,:,1:].sum(axis=1)
        return labels
    




def _stack_batch(data_batch):
    """Stacks batches of input, output and weight tensors into a larger batch (with padding)"""

    max_sequence_length = max([len(labels[0]) for _,labels,_ in data_batch])
    
    # Create arrays for each input one by one
    all_inputs = []
    nb_inputs = len(data_batch[0][0])
    for i in range(nb_inputs):
        if len(data_batch[0][0][i].shape) == 2:
            all_inputs_part = np.zeros((len(data_batch), max_sequence_length))
        else:
            all_inputs_part = np.zeros((len(data_batch), max_sequence_length,
                                        data_batch[0][0][i].shape[-1]))  
        for j, (inputs_in_batch, _,_) in enumerate(data_batch):
            all_inputs_part[j,:len(inputs_in_batch[i][0])] = inputs_in_batch[i][0]
        all_inputs.append(all_inputs_part)
        
    all_targets = np.zeros((len(data_batch), max_sequence_length, data_batch[0][1].shape[-1]), dtype=bool)
    
    # The default weight is near-zero to ignore padded values during training and evaluation 
    all_weights = 0.0001 * np.ones((len(data_batch), max_sequence_length), dtype=np.float32)
    
    for j, (_, outputs_in_batch, weights_in_batch) in enumerate(data_batch):
        all_targets[j,:len(outputs_in_batch[0])] = outputs_in_batch[0]
        all_weights[j,:len(weights_in_batch[0])] = weights_in_batch[0] 
        
    return all_inputs, all_targets, all_weights

        

def generate_from_docbin(docbin_file, target_source=None, cutoff=None, nb_to_skip=0, 
                         labels_to_retain=None, labels_to_map=None, loop=False):
    """Generates spacy documents from a DocBin object."""
    
    import annotations
    
    nb_generated = 0
    vocab = spacy.load("en_core_web_md").vocab
    
    while True:         
        reader = annotations.docbin_reader(docbin_file, vocab=vocab, cutoff=cutoff, nb_to_skip=nb_to_skip)
        for spacy_doc in reader:
            
            spans = []
            if target_source is None:
                spans = [(ent.start,ent.end, ent.label_) for ent in spacy_doc.ents]
            else:
                spans = [(start,end, label) 
                         for (start,end), vals in spacy_doc.user_data["annotations"][target_source].items() 
                         for label, conf in vals if conf > 0.5]
                 
            new_spans = []
            for start,end,label in spans:
                if labels_to_map is not None:
                    label = labels_to_map.get(label, label)
                if labels_to_retain is None or label in labels_to_retain:
                    ent = spacy.tokens.Span(spacy_doc, start, end, spacy_doc.vocab.strings[label])
                    new_spans.append(ent)
            spacy_doc.ents = tuple(new_spans)
             
            yield spacy_doc
            nb_generated += 1
            if cutoff is not None and nb_generated >= cutoff:
                return
        if not loop:
            break
    
                      
                    
