import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sparc_model import SQLBart
from sparc_dataset import SparcDataModule
from transformers import BartTokenizer


if __name__ == '__main__':
    config_name = 'facebook/bart-base'
    bart_tokenizer = BartTokenizer.from_pretrained(config_name)
    sql_bart = SQLBart(bart_tokenizer)

    sparc_data = SparcDataModule('sparc/', batch_size=2, tokenizer=bart_tokenizer)
    # trainer = pl.Trainer(gpus=-1, precision=16, default_root_dir='checkpoints',
    #                      terminate_on_nan=True, accumulate_grad_batches=1,
    #                      # check_val_every_n_epoch=20,
    #                      gradient_clip_val=5,
    #                      callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='min')],
    #                      )
    trainer = pl.Trainer(default_root_dir='checkpoints',
                         terminate_on_nan=True, accumulate_grad_batches=1,
                         check_val_every_n_epoch=5,
                         gradient_clip_val=5, gradient_clip_algorithm='value',
                         callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='min')],
                         )
    trainer.fit(model=sql_bart, datamodule=sparc_data)

    trainer.test(model=sql_bart, datamodule=sparc_data)
