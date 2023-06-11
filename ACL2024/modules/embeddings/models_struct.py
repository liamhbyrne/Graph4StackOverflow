"""
Models Struct
Author: Liam Byrne
"""
import spacy
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel

from ACL2024.modules.embeddings.module_embedding import ModuleEmbeddingTrainer
from ACL2024.modules.embeddings.tag_embedding import NextTagEmbeddingTrainer
from ACL2024.modules.util.unixcoder import UniXcoder


class ModelStore:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    en = spacy.load('en_core_web_sm')
    stopwords = en.Defaults.stop_words
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).to(device)
    unixcoder = UniXcoder("microsoft/unixcoder-base").to(device)
    codebert_tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    codebert_model = AutoModel.from_pretrained('microsoft/codebert-base')
    tag_embedding_model = NextTagEmbeddingTrainer.load_model("../embeddings/pre-trained/tag-emb-7_5mil-50d-63653-3.pt", embedding_dim=50, vocab_size=63654, context_length=3)
    module_embedding_model = ModuleEmbeddingTrainer.load_model("../embeddings/pre-trained/module-emb-1milx5-30d-49911.pt", embedding_dim=30, vocab_size=49911)

if __name__ == '__main__':
    ms = ModelStore()
    print(ms.en)
    print(ms.stopwords)
    print(ms.bert_tokenizer)
    print(ms.bert_model)
    print(ms.unixcoder)
    print(ms.codebert_tokenizer)
    print(ms.codebert_model)
    print(ms.tag_embedding_model)
    print(ms.module_embedding_model)