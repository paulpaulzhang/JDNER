import os
from ark_nlp.model.ner.global_pointer_bert import Task
import torch
from utils import FGM, PGD, get_evaluate_fpr, Logs
from tqdm import tqdm


class MyGlobalPointerNERTask(Task):

    def fit(
        self,
        args,
        train_data,
        validation_data=None,
        lr=False,
        params=None,
        batch_size=32,
        epochs=1,
        gradient_accumulation_steps=1,
        save_each_model=True,
        **kwargs
    ):
        """
        训练方法

        Args:
            train_data (:obj:`ark_nlp dataset`): 训练的batch文本
            validation_data (:obj:`ark_nlp dataset`): 验证的batch文本
            lr (:obj:`float` or :obj:`bool`, optional, defaults to False): 学习率
            params (:obj:`str` or :obj:`torch.optim.Optimizer` or :obj:`list` or :obj:`None`, optional, defaults to None): 优化器，可能是名称、对象、参数列表
            batch_size (:obj:`int`, optional, defaults to 32): batch大小
            epochs (:obj:`int`, optional, defaults to 1): 训练轮数
            gradient_accumulation_steps (:obj:`int`, optional, defaults to 1): 梯度累计数
            **kwargs (optional): 其他可选参数
        """  # noqa: ignore flake8"

        self.logs = dict()

        train_generator = self._on_train_begin(
            train_data,
            validation_data,
            batch_size,
            lr,
            params,
            shuffle=True,
            **kwargs
        )

        ckpt = os.path.join(
            args.checkpoint, args.model_type)
        os.makedirs(ckpt, exist_ok=True)
        logs = Logs(os.path.join(ckpt, 'log.txt'))
        logs.write(str(args) + '\n')
        logs.write(
            f"|{'epoch':^15}|{'loss':^15}|{'precision':^15}|{'recall':^15}|{'f1':^15}|{'best model':^15}|\n")

        best_f1 = 0
        for epoch in range(1, epochs+1):

            self._on_epoch_begin(**kwargs)

            train_iterator = tqdm(
                train_generator, desc=f'Epoch : {epoch}', total=len(train_generator))

            for step, inputs in enumerate(train_iterator):

                self._on_step_begin(epoch, step, inputs, **kwargs)

                # input处理和设备转移
                inputs = self._get_module_inputs_on_train(inputs, **kwargs)

                outputs = self.module(**inputs)
                logits, loss = self._get_train_loss(inputs, outputs, **kwargs)

                # loss backword
                loss = self._on_backward(
                    inputs, outputs, logits, loss, **kwargs)

                if args.use_fgm:
                    self.module.zero_grad()
                    fgm = FGM(args, self.module)
                    fgm.attack()
                    outputs = self.module(**inputs)
                    adv_logits, adv_loss = self._get_train_loss(
                        inputs, outputs, **kwargs)
                    adv_loss = self._on_backward(
                        inputs, outputs, adv_logits, adv_loss, **kwargs)
                    fgm.restore()

                if args.use_pgd:
                    self.module.zero_grad()
                    pgd = PGD(args, self.module)
                    pgd.backup_grad()
                    for t in range(args.adv_k):
                        pgd.attack(is_first_attack=(t == 0))
                        if t != args.adv_k - 1:
                            self.module.zero_grad()
                        else:
                            pgd.restore_grad()
                        outputs = self.module(**inputs)
                        adv_logits, adv_loss = self._get_train_loss(
                            inputs, outputs, **kwargs)
                        adv_loss = self._on_backward(
                            inputs, outputs, adv_logits, adv_loss, **kwargs)
                    pgd.restore()

                if (step + 1) % gradient_accumulation_steps == 0:

                    # optimize
                    self._on_optimize(inputs, outputs, logits, loss, **kwargs)

                # setp evaluate
                self._on_step_end(step, inputs, outputs,
                                  loss, verbose=False, **kwargs)
                train_iterator.set_postfix_str(
                    f"training loss: {(self.logs['epoch_loss'] / self.logs['epoch_step']):.4f}")

            self._on_epoch_end(epoch, verbose=False, **kwargs)

            if save_each_model:
                torch.save(self.module.state_dict(),
                           os.path.join(ckpt, f'epoch{epoch}.pth'))

            if validation_data is not None:
                self.evaluate(validation_data, ** kwargs)
                content = "|{:^15}|{:^15}|{:^15}|{:^15}|{:^15}|{:^15}|\n".format(
                    epoch,
                    round(self.evaluate_logs['eval_loss'] /
                          self.evaluate_logs['eval_step'], 5),
                    round(self.evaluate_logs['precision'], 5), round(
                        self.evaluate_logs['recall'], 5),
                    round(self.evaluate_logs['f1'], 5), 'True' if self.evaluate_logs['f1'] > best_f1 else '')
                logs.write(content)
                if self.evaluate_logs['f1'] > best_f1:
                    best_f1 = self.evaluate_logs['f1']
                    torch.save(self.module.state_dict(),
                               os.path.join(ckpt, f'best_model.pth'))
                else:
                    break

        if not save_each_model:
            torch.save(self.module.state_dict(),
                       os.path.join(ckpt, f'last_model.pth'))

        self._on_train_end(**kwargs)

    def _on_evaluate_begin_record(self, **kwargs):

        self.evaluate_logs['eval_loss'] = 0
        self.evaluate_logs['eval_step'] = 0
        self.evaluate_logs['eval_example'] = 0

        self.evaluate_logs['labels'] = []
        self.evaluate_logs['logits'] = []
        self.evaluate_logs['input_lengths'] = []

        self.evaluate_logs['tp'] = 0
        self.evaluate_logs['fp'] = 0
        self.evaluate_logs['fn'] = 0

        self.evaluate_logs['f1'] = 0
        self.evaluate_logs['precision'] = 0
        self.evaluate_logs['recall'] = 0

    def _on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():

            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)

            tp, fp, fn = get_evaluate_fpr(
                inputs['label_ids'].to_dense(),
                logits
            )
            self.evaluate_logs['tp'] += tp
            self.evaluate_logs['fp'] += fp
            self.evaluate_logs['fn'] += fn

        self.evaluate_logs['eval_example'] += len(inputs['label_ids'])
        self.evaluate_logs['eval_step'] += 1
        self.evaluate_logs['eval_loss'] += loss.item()

    def _on_evaluate_epoch_end(
        self,
        validation_data,
        epoch=1,
        is_evaluate_print=True,
        id2cat=None,
        **kwargs
    ):

        if id2cat is None:
            id2cat = self.id2cat

        if is_evaluate_print:
            precision = self.evaluate_logs['tp'] / \
                (self.evaluate_logs['tp'] + self.evaluate_logs['fp'])
            recall = self.evaluate_logs['tp'] / \
                (self.evaluate_logs['tp'] + self.evaluate_logs['fn'])
            f1 = 2 * precision * recall / (precision + recall)
            self.evaluate_logs['f1'] = f1
            self.evaluate_logs['precision'] = precision
            self.evaluate_logs['recall'] = recall
            print('eval loss: {:.6f}, precision: {:.6f}, recall: {:.6f}, f1_score: {:.6f}\n'.format(
                self.evaluate_logs['eval_loss'] /
                self.evaluate_logs['eval_step'],
                precision, recall, f1))
