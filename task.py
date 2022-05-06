from ast import arg
import math
import os
import warnings
from ark_nlp.factory.task.base._sequence_classification import SequenceClassificationTask
from utils import FGM, PGD
from torch.utils.data import DataLoader
from ark_nlp.factory.optimizer import get_optimizer
import torch
from utils import Logs
from tqdm import tqdm
from utils import metrics


class Task(SequenceClassificationTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_f1 = 0

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
            args=args,
            **kwargs
        )

        ckpt = os.path.join(
            args.checkpoint, args.model_type)
        os.makedirs(ckpt, exist_ok=True)
        logs = Logs(os.path.join(ckpt, 'log.txt'))
        for k, v in vars(args).items():
            logs.write(f'{k}: {v}' + '\n')
        logs.write(
            f"|{'steps':^15}|{'loss':^15}|{'precision':^15}|{'recall':^15}|{'f1':^15}|\n")

        one_epoch = int(math.ceil(len(train_generator) /
                        gradient_accumulation_steps))  # 将step转换为epoch
        total_steps = epochs * one_epoch
        cur_steps = 0

        args.early_stopping = int(
            math.ceil(args.early_stopping * one_epoch / args.eval_steps))
        early_stopping = args.early_stopping

        if args.training_strategy == 'epoch':
            args.eval_steps = one_epoch

        progress_bar = tqdm(range(total_steps))
        for epoch in range(1, epochs+1):

            self._on_epoch_begin(**kwargs)

            for step, inputs in enumerate(train_generator):

                self._on_step_begin(epoch, step, inputs, **kwargs)

                # input处理和设备转移
                inputs = self._get_module_inputs_on_train(inputs, **kwargs)

                outputs = self.module(**inputs)
                logits, loss = self._get_train_loss(inputs, outputs, **kwargs)

                # loss backword
                loss = self._on_backward(
                    inputs, outputs, logits, loss, args=args, **kwargs)

                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_generator):

                    # optimize
                    self._on_optimize(inputs, outputs, logits, loss, **kwargs)

                    progress_bar.set_postfix_str(
                        "epoch: {:.2f} loss: {:.4f}".
                        format(cur_steps / len(train_generator),
                               (self.logs['epoch_loss'] / self.logs['epoch_step'])))

                    progress_bar.update(1)
                    cur_steps += 1

                    if cur_steps % args.eval_steps == 0 or (step + 1) == len(train_generator):
                        if validation_data is not None:
                            self.evaluate(validation_data,
                                          ckpt=ckpt, ** kwargs)

                            content = "|{:^15}|{:^15}|{:^15}|{:^15}|{:^15}|\n".format(
                                cur_steps,
                                round(self.evaluate_logs['eval_loss'] /
                                      self.evaluate_logs['eval_step'], 3),
                                round(self.evaluate_logs['precision'], 3), round(
                                    self.evaluate_logs['recall'], 3),
                                round(self.evaluate_logs['f1'], 5))
                            logs.write(content)

                            if self.evaluate_logs['f1'] < self.best_f1:
                                early_stopping -= 1
                            else:
                                early_stopping = args.early_stopping

                            if early_stopping == 0:
                                self._on_train_end(**kwargs)
                                return

                            self.module.train()

        print('best f1:', self.best_f1)
        self._on_train_end(ckpt=ckpt, **kwargs)

    def _on_train_begin(
        self,
        train_data,
        validation_data,
        batch_size,
        lr,
        params,
        shuffle,
        num_workers=0,
        train_to_device_cols=None,
        args=None,
        **kwargs
    ):
        if hasattr(train_data, 'id2cat'):
            self.id2cat = train_data.id2cat
            self.cat2id = {v_: k_ for k_, v_ in train_data.id2cat.items()}

        # 在初始化时会有class_num参数，若在初始化时不指定，则在训练阶段从训练集获取信息
        if self.class_num is None:
            if hasattr(train_data, 'class_num'):
                self.class_num = train_data.class_num
            else:
                warnings.warn("The class_num is None.")

        if train_to_device_cols is None:
            self.train_to_device_cols = train_data.to_device_cols
        else:
            self.train_to_device_cols = train_to_device_cols

        train_generator = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._train_collate_fn
        )
        self.train_generator_lenth = len(train_generator)

        self.optimizer = get_optimizer(self.optimizer, self.module, lr, params)
        self.optimizer.zero_grad()

        self.module.train()

        if args.use_fgm:
            self.fgm = FGM(self.module)
        if args.use_pgd:
            self.pgd = PGD(self.module)

        self._on_train_begin_record(**kwargs)

        return train_generator

    def _on_train_end(self, ckpt=None, save_last_model=False, **kwargs):
        if save_last_model:
            if self.ema_decay:
                self.ema.store(self.module.parameters())
                self.ema.copy_to(self.module.parameters())

            state_dict = {k: v for k, v in self.module.state_dict(
            ).items() if 'relative_positions' not in k}
            torch.save(state_dict, os.path.join(ckpt, f'last_model.pth'))

            if self.ema_decay:
                self.ema.restore(self.module.parameters())

    def _on_backward(
        self,
        inputs,
        outputs,
        logits,
        loss,
        gradient_accumulation_steps=1,
        args=None,
        **kwargs
    ):

        # 如果GPU数量大于1
        if self.n_gpu > 1:
            loss = loss.mean()
        # 如果使用了梯度累积，除以累积的轮数
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        if args.use_fgm:
            self.fgm.attack(epsilon=args.epsilon, emb_name=args.emb_name)
            logits = self.module(**inputs)
            _, attck_loss = self._get_train_loss(inputs, logits, **kwargs)
            attck_loss.backward()
            self.fgm.restore()

        if args.use_pgd:
            self.pgd.backup_grad()
            for t in range(args.adv_k):
                self.pgd.attack(is_first_attack=(t == 0))
                if t != args.adv_k - 1:
                    self.optimizer.zero_grad()
                else:
                    self.pgd.restore_grad()
                logits = self.module(**inputs)
                _, attck_loss = self._get_train_loss(inputs, logits, **kwargs)
                attck_loss.backward()
            self.pgd.restore()

        self._on_backward_record(loss, **kwargs)

        return loss

    def _on_evaluate_begin_record(self, **kwargs):

        self.evaluate_logs['eval_loss'] = 0
        self.evaluate_logs['eval_step'] = 0
        self.evaluate_logs['eval_example'] = 0

        self.evaluate_logs['labels'] = []
        self.evaluate_logs['logits'] = []
        self.evaluate_logs['input_lengths'] = []

        self.evaluate_logs['y_true'] = []
        self.evaluate_logs['y_pred'] = []

        self.evaluate_logs['f1'] = 0
        self.evaluate_logs['precision'] = 0
        self.evaluate_logs['recall'] = 0

    def _on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():

            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)

            self.evaluate_logs['y_true'] += inputs['label_ids'].cpu().numpy().tolist()
            self.evaluate_logs['y_pred'] += torch.argmax(
                logits, dim=1).cpu().numpy().tolist()

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
            f1, precision, recall = metrics(
                self.evaluate_logs['y_true'], self.evaluate_logs['y_pred'])
            self.evaluate_logs['f1'] = f1
            self.evaluate_logs['precision'] = precision
            self.evaluate_logs['recall'] = recall
            print('\neval loss: {:.3f}, precision: {:.4f}, recall: {:.4f}, f1_score: {:.5f}\n'.format(
                self.evaluate_logs['eval_loss'] /
                self.evaluate_logs['eval_step'],
                precision, recall, f1))

    def _on_evaluate_end(self, ckpt=None, **kwargs):

        self._on_evaluate_end_record()

        if self.evaluate_logs['f1'] > self.best_f1 and ckpt:
            self.best_f1 = self.evaluate_logs['f1']
            state_dict = {k: v for k, v in self.module.state_dict(
            ).items() if 'relative_positions' not in k}
            torch.save(state_dict, os.path.join(ckpt, f'best_model.pth'))

        if self.ema_decay:
            self.ema.restore(self.module.parameters())
