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
                              hidden_size=config.hidden_size //  2,
                              bidirectional=True,
                              dropout=0.1,
                              batch_first=True)

        for param in self.encoder.parameters():
            param.requires_grad = encoder_trained

        self.global_pointer = GlobalPointer(
            self.num_labels,
            head_size,
            config.hidden_size
        )

        # self.global_pointer1 = GlobalPointer(
        #     self.num_labels,
        #     head_size,
        #     config.hidden_size
        # )

        # self.global_pointer2 = GlobalPointer(
        #     self.num_labels,
        #     head_size,
        #     config.hidden_size
        # )

        # self.global_pointer3 = GlobalPointer(
        #     self.num_labels,
        #     head_size,
        #     config.hidden_size
        # )

        # self.global_pointer4 = GlobalPointer(
        #     self.num_labels,
        #     head_size,
        #     config.hidden_size
        # )

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
        # logits = self.global_pointer(sequence_output, mask=attention_mask)

        output, (h_n, c_n) = self.bilstm(sequence_output)

        # all_hidden_states = outputs[2]
        # sequence_output_last_1 = all_hidden_states[-1]
        # sequence_output_last_2 = all_hidden_states[-2]
        # sequence_output_last_3 = all_hidden_states[-3]
        # sequence_output_last_4 = all_hidden_states[-4]

        # logits1 = self.global_pointer1(sequence_output_last_1, mask=attention_mask)
        # logits2 = self.global_pointer2(sequence_output_last_2, mask=attention_mask)
        # logits3 = self.global_pointer3(sequence_output_last_3, mask=attention_mask)
        # logits4 = self.global_pointer4(sequence_output_last_4, mask=attention_mask)

        # logits = logits1 + logits2 + logits3 + logits4
        logits = self.global_pointer(output, mask=attention_mask)
        return logits


class GlobalPointerEnsambleModel(nn.Module):

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

        for param in self.encoder.parameters():
            param.requires_grad = encoder_trained

        self.gps = [GlobalPointer(
            self.num_labels, head_size, config.hidden_size) for _ in range(10)]

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

        logits = []
        for gp in self.gps:
            logits.append(gp(sequence_output, mask=attention_mask))

        logits = torch.sum(torch.stack(logits), dim=0)

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
