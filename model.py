from ark_nlp.nn.layer.global_pointer_block import GlobalPointer, EfficientGlobalPointer
from torch import nn
import torch


class GlobalPointerModel(nn.Module):

    def __init__(
        self,
        config,
        encoder,
        encoder_trained=True,
        head_size=64
    ):
        super(GlobalPointerModel, self).__init__()

        self.num_labels = config.num_labels

        self.encoder = encoder

        self.bilstm = nn.LSTM(input_size=config.hidden_size,
                              hidden_size=config.hidden_size,
                              bidirectional=True,
                              dropout=0.1,
                              batch_first=True)

        for param in self.encoder.parameters():
            param.requires_grad = encoder_trained

        self.global_pointer = GlobalPointer(
            self.num_labels,
            head_size * 2,
            config.hidden_size * 2
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs
    ):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]

        # 待测试效果
        output, (h_n, c_n) = self.bilstm(sequence_output)

        # all_hidden_state = outputs[2]
        # sequence_output = all_hidden_state[-1] * 0.4 + \
        #     all_hidden_state[-2] * 0.3 + all_hidden_state[-3] * 0.3

        logits = self.global_pointer(output, mask=attention_mask)

        return logits


class EfficientGlobalPointerModel(nn.Module):

    def __init__(
        self,
        config,
        encoder,
        encoder_trained=True,
        head_size=64
    ):
        super(EfficientGlobalPointerModel, self).__init__()

        self.num_labels = config.num_labels

        self.encoder = encoder

        for param in self.encoder.parameters():
            param.requires_grad = encoder_trained

        self.efficient_global_pointer = EfficientGlobalPointer(
            self.num_labels,
            head_size,
            config.hidden_size
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs
    ):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]

        logits = self.efficient_global_pointer(
            sequence_output, mask=attention_mask)

        return logits