from transformers import LongformerForSequenceClassification, LongformerModel, AutoModel
import numpy as np
import torch


def sinusoidal_init(num_embeddings: int, embedding_dim: int):
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / embedding_dim) for i in range(embedding_dim)]
        if pos != 0 else np.zeros(embedding_dim) for pos in range(num_embeddings)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class LongformerClassifier(LongformerForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.num_labels
        
    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        global_attention_mask = x[:, :, 2]

        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )[0]
        outputs = outputs.squeeze(-1)
        return outputs


class LongformerFeaturizer(LongformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        global_attention_mask = x[:, :, 2]

        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )[1]  # get pooled output
        return outputs


class HierBERTClassifier(torch.nn.Module):
    def __init__(self, config, d_out):
        super().__init__()
        self.d_out = d_out

        # Specs for the segment-wise encoder
        self.bert_encoder = AutoModel.from_pretrained(config.model)
        self.hidden_size = self.bert_encoder.config.hidden_size
        self.max_segments = config.max_segments
        self.max_segment_length = config.max_segment_length

        # Init sinusoidal positional embeddings
        self.seg_pos_embeddings = torch.nn.Embedding(config.max_segments + 1, self.bert_encoder.config.hidden_size,
                                                     padding_idx=0,
                                                     _weight=sinusoidal_init(config.max_segments + 1,
                                                                             self.bert_encoder.config.hidden_size))
        # Init segment-wise transformer-based encoder
        self.seg_encoder = torch.nn.Transformer(d_model=self.bert_encoder.config.hidden_size,
                                                nhead=self.bert_encoder.config.num_attention_heads,
                                                batch_first=True,
                                                dim_feedforward=self.bert_encoder.config.intermediate_size,
                                                activation=self.bert_encoder.config.hidden_act,
                                                dropout=self.bert_encoder.config.hidden_dropout_prob,
                                                layer_norm_eps=self.bert_encoder.config.layer_norm_eps,
                                                num_encoder_layers=2, num_decoder_layers=0).encoder

        # Init Classifier
        self.dropout = torch.nn.Dropout(self.bert_encoder.config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.hidden_size, self.d_out)

    def __call__(self, x):
        input_ids = x[:, :, :, 0]
        attention_mask = x[:, :, :, 1]

        input_ids_reshape = input_ids.contiguous().view(-1, input_ids.size(-1))
        attention_mask_reshape = attention_mask.contiguous().view(-1, attention_mask.size(-1))

        encoder_outputs = self.bert_encoder(input_ids=input_ids_reshape,
                                            attention_mask=attention_mask_reshape,
                                            )[0]

        # Reshape back to (batch_size, n_segments, max_segment_length, output_size) --> (4, 64, 128, 768)
        encoder_outputs = encoder_outputs.contiguous().view(input_ids.size(0), self.max_segments,
                                                            self.max_segment_length,
                                                            self.hidden_size)

        # Gather CLS outputs per segment --> (4, 64, 768)
        encoder_outputs = encoder_outputs[:, :, 0]

        # Infer real segments, i.e., mask paddings
        seg_mask = (torch.sum(input_ids, 2) != 0).to(input_ids.dtype)
        # Infer and collect segment positional embeddings
        seg_positions = torch.arange(1, self.max_segments + 1).to(input_ids.device) * seg_mask
        # Add segment positional embeddings to segment inputs
        encoder_outputs += self.seg_pos_embeddings(seg_positions)
        # Encode segments with segment-wise transformer
        seg_encoder_outputs = self.seg_encoder(encoder_outputs)

        # Collect document representation
        pooled_output, _ = torch.max(seg_encoder_outputs, 1)

        # Use classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class HierBERTFeaturizer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        # Specs for the segment-wise encoder
        self.bert_encoder = AutoModel.from_pretrained(config.model)
        self.hidden_size = self.bert_encoder.config.hidden_size
        self.d_out = self.hidden_size
        self.max_segments = config.max_segments
        self.max_segment_length = config.max_segment_length

        # Init sinusoidal positional embeddings
        self.seg_pos_embeddings = torch.nn.Embedding(config.max_segments + 1, self.bert_encoder.config.hidden_size,
                                                     padding_idx=0,
                                                     _weight=sinusoidal_init(config.max_segments + 1,
                                                                             self.bert_encoder.config.hidden_size))
        # Init segment-wise transformer-based encoder
        self.seg_encoder = torch.nn.Transformer(d_model=self.bert_encoder.config.hidden_size,
                                                nhead=self.bert_encoder.config.num_attention_heads,
                                                batch_first=True,
                                                dim_feedforward=self.bert_encoder.config.intermediate_size,
                                                activation=self.bert_encoder.config.hidden_act,
                                                dropout=self.bert_encoder.config.hidden_dropout_prob,
                                                layer_norm_eps=self.bert_encoder.config.layer_norm_eps,
                                                num_encoder_layers=2, num_decoder_layers=0).encoder

    def __call__(self, x):
        input_ids = x[:, :, :, 0]
        attention_mask = x[:, :, :, 1]

        input_ids_reshape = input_ids.contiguous().view(-1, input_ids.size(-1))
        attention_mask_reshape = attention_mask.contiguous().view(-1, attention_mask.size(-1))

        encoder_outputs = self.bert_encoder(input_ids=input_ids_reshape,
                                            attention_mask=attention_mask_reshape,
                                            )[0]

        # Reshape back to (batch_size, n_segments, max_segment_length, output_size) --> (4, 64, 128, 768)
        encoder_outputs = encoder_outputs.contiguous().view(input_ids.size(0), self.max_segments,
                                                            self.max_segment_length,
                                                            self.hidden_size)

        # Gather CLS outputs per segment --> (4, 64, 768)
        encoder_outputs = encoder_outputs[:, :, 0]

        # Infer real segments, i.e., mask paddings
        seg_mask = (torch.sum(input_ids, 2) != 0).to(input_ids.dtype)
        # Infer and collect segment positional embeddings
        seg_positions = torch.arange(1, self.max_segments + 1).to(input_ids.device) * seg_mask
        # Add segment positional embeddings to segment inputs
        encoder_outputs += self.seg_pos_embeddings(seg_positions)

        # Encode segments with segment-wise transformer
        seg_encoder_outputs = self.seg_encoder(encoder_outputs)

        # Collect document representation
        pooled_output, _ = torch.max(seg_encoder_outputs, 1)

        return pooled_output


