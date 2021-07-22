import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from t5_model import SQLT5
from spider_dataset import SpiderDataModule
from transformers import T5Tokenizer


if __name__ == '__main__':
    config_name = 'dbernsohn/t5_wikisql_SQL2en'
    t5_tokenizer = T5Tokenizer.from_pretrained(config_name)
    sql_t5 = SQLT5(t5_tokenizer)

    sparc_data = SpiderDataModule('spider/', batch_size=32, tokenizer=t5_tokenizer)
    trainer = pl.Trainer(gpus=-1, precision=16, default_root_dir='checkpoints',
                         terminate_on_nan=True, accumulate_grad_batches=1,
                         gradient_clip_val=5,
                         callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='min')],
                         )
    # trainer = pl.Trainer(default_root_dir='checkpoints',
    #                      terminate_on_nan=True, accumulate_grad_batches=1,
    #                      gradient_clip_val=5, gradient_clip_algorithm='value',
    #                      callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='min')],
    #                      )
    trainer.fit(model=sql_t5, datamodule=sparc_data)

    trainer.test(model=sql_t5, datamodule=sparc_data)
