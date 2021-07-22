import os
import re
import json
import attr
import copy
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import networkx as nx
import pytorch_lightning as pl

import evaluation
from transformers import BartTokenizer


SQL_RESERVE_TOKENS = ['select', 'from', 'where', 'group by', 'having', 'order by', 'desc', 'asc',
                      'distinct', 'and', 'or', 'not', 'between', 'not in', 'in', 'like', 'not exists', 'exists',
                      'as', 'with', 'union', 'intersect', 'except', 'join', 'limit']
SQL_RESERVE_AGGRS = ['max', 'min', 'avg', 'sum', 'count']


@attr.s
class SparcItem:
    id = attr.ib()
    text = attr.ib()
    code = attr.ib()
    schema = attr.ib()
    orig = attr.ib()
    orig_schema = attr.ib()
    db_name = attr.ib()


@attr.s
class Column:
    id = attr.ib()
    table = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    type = attr.ib()
    foreign_key_for = attr.ib(default=None)


@attr.s
class Table:
    id = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    columns = attr.ib(factory=list)
    primary_keys = attr.ib(factory=list)


@attr.s
class Schema:
    db_id = attr.ib()
    tables = attr.ib()
    columns = attr.ib()
    foreign_key_graph = attr.ib()
    orig = attr.ib()


def load_tables(path):
    schemas = {}
    eval_foreign_key_maps = {}

    schema_dicts = json.load(open(path))
    for schema_dict in schema_dicts:
        tables = tuple(
            Table(
                id=i,
                name=name.split(),
                unsplit_name=name,
                orig_name=orig_name,
            )
            for i, (name, orig_name) in enumerate(zip(
                schema_dict['table_names'], schema_dict['table_names_original']))
        )
        columns = tuple(
            Column(
                id=i,
                table=tables[table_id] if table_id >= 0 else None,
                name=col_name.split(),
                unsplit_name=col_name,
                orig_name=orig_col_name,
                type=col_type,
            )
            for i, ((table_id, col_name), (_, orig_col_name), col_type) in enumerate(zip(
                schema_dict['column_names'],
                schema_dict['column_names_original'],
                schema_dict['column_types']))
        )

        # Link columns to tables
        for column in columns:
            if column.table:
                column.table.columns.append(column)

        for column_id in schema_dict['primary_keys']:
            # Register primary keys
            column = columns[column_id]
            column.table.primary_keys.append(column)

        foreign_key_graph = nx.DiGraph()
        for source_column_id, dest_column_id in schema_dict['foreign_keys']:
            # Register foreign keys
            source_column = columns[source_column_id]
            dest_column = columns[dest_column_id]
            source_column.foreign_key_for = dest_column
            foreign_key_graph.add_edge(
                source_column.table.id,
                dest_column.table.id,
                columns=(source_column_id, dest_column_id))
            foreign_key_graph.add_edge(
                dest_column.table.id,
                source_column.table.id,
                columns=(dest_column_id, source_column_id))

        db_id = schema_dict['db_id']
        assert db_id not in schemas
        schemas[db_id] = Schema(db_id, tables, columns, foreign_key_graph, schema_dict)
        eval_foreign_key_maps[db_id] = evaluation.build_foreign_key_map(schema_dict)

    return schemas, eval_foreign_key_maps


class SparcDataset(torch.utils.data.Dataset):
    def __init__(self, path, tables_paths, db_path, tokenizer, mode='train', limit=None):
        self.path = path
        self.db_path = db_path
        self.examples = []
        self.use_column_type = False
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_seq_len = self.tokenizer.model_max_length

        self.schemas, self.eval_foreign_key_maps = load_tables(tables_paths)

        raw_data = json.load(open(path))
        for entry in tqdm(raw_data):
            accumulated_toks = []
            for i, interaction in enumerate(entry['interaction']):
                new_toks = interaction['utterance_toks']
                accumulated_toks.append(new_toks)
                item = SparcItem(
                    id=len(self.examples),
                    text=copy.deepcopy(accumulated_toks),
                    code=interaction['query'],
                    schema=self.schemas[entry['database_id']],
                    orig=(entry, i),
                    orig_schema=self.schemas[entry['database_id']].orig,
                    db_name=entry['database_id']
                )
                if self.validate_item(item):
                    self.examples.append(item)
            if self.mode == 'train':
                item = SparcItem(
                    id=len(self.examples),
                    text=[entry['final']['utterance'].split()],
                    code=entry['final']['query'],
                    schema=self.schemas[entry['database_id']],
                    orig=(entry, -1),
                    orig_schema=self.schemas[entry['database_id']].orig,
                    db_name=entry['database_id']
                )
                if self.validate_item(item):
                    self.examples.append(item)
            # if len(self.examples) >= 5000: break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        encoder_dict, decoder_dict = self.tokenize_item(item)
        return {
            'id': item.id,
            'input_ids': encoder_dict['input_ids'],
            'attention_mask': encoder_dict['attention_mask'],
            'decoder_input_ids': decoder_dict['input_ids'],
            'decoder_attention_mask': decoder_dict['attention_mask'],
            'labels': copy.deepcopy(decoder_dict['input_ids']),
            'db_name': self.tokenizer(item.db_name)['input_ids']
        }

    def tokenize_item(self, item):
        nl = ' '.join([t for s in item.text for t in s])
        if self.mode != 'test':
            sql = item.code.replace("'", '"')
            sql = ' ' + re.sub(r'\b(?<!")(\w+)(?!")\b', lambda match: match.group(1).lower(), sql) + ' '
            for token in SQL_RESERVE_TOKENS:
                sql = sql.replace(' ' + token + ' ', ' ' + token.upper() + ' ')
            for token in SQL_RESERVE_AGGRS:
                sql = sql.replace(token + '(', token.upper() + '(')
            sql = sql.strip().replace('  ', ' ')
            # sql = sql.replace(' ', '<space>')
        else:
            sql = ''
        columns = []
        for c in item.schema.columns:
            if c and c.table:
                tn, cn = c.table.orig_name, c.orig_name
                columns.append((tn, cn))
        concat_input = nl + self.tokenizer.eos_token
        for c in columns:
            concat_input += ' <c>' + c[0].lower() + '</s>' + c[1].lower()
        encoder_dict = self.tokenizer(concat_input)
        decoder_dict = self.tokenizer(sql)
        return encoder_dict, decoder_dict

    def validate_item(self, item):
        encoder_dict, decoder_dict = self.tokenize_item(item)
        return len(encoder_dict['input_ids']) < self.max_seq_len and len(decoder_dict['input_ids']) < self.max_seq_len

    @staticmethod
    def collate_fn(x_list):
        max_input_len = max(len(x['input_ids']) for x in x_list)
        max_output_len = max(len(x['decoder_input_ids']) for x in x_list)
        max_dbname_len = max(len(x['db_name']) for x in x_list)
        for x in x_list:
            x['input_ids'] += [0 for _ in range(max_input_len - len(x['input_ids']))]
            x['attention_mask'] += [0 for _ in range(max_input_len - len(x['attention_mask']))]
            x['decoder_input_ids'] += [0 for _ in range(max_output_len - len(x['decoder_input_ids']))]
            x['decoder_attention_mask'] += [0 for _ in range(max_output_len - len(x['decoder_attention_mask']))]
            x['labels'] += [-100 for _ in range(max_output_len - len(x['labels']))]
            x['db_name'] += [0 for _ in range(max_dbname_len - len(x['db_name']))]
        return default_collate([{k: torch.tensor(v).long() for k, v in x.items()} for x in x_list])

    class Metrics:
        def __init__(self, dataset):
            self.dataset = dataset
            self.foreign_key_maps = {
                db_id: evaluation.build_foreign_key_map(schema.orig)
                for db_id, schema in self.dataset.schemas.items()
            }
            self.evaluator = evaluation.Evaluator(
                self.dataset.db_path,
                self.foreign_key_maps,
                'match')
            self.results = []

        def add(self, item, inferred_code, orig_question=None):
            ret_dict = self.evaluator.evaluate_one(
                item.schema.db_id, item.orig['query'], inferred_code)
            if orig_question:
                ret_dict["orig_question"] = orig_question
            self.results.append(ret_dict)

        def add_beams(self, item, inferred_codes, orig_question=None):
            beam_dict = {}
            if orig_question:
                beam_dict["orig_question"] = orig_question
            for i, code in enumerate(inferred_codes):
                ret_dict = self.evaluator.evaluate_one(
                    item.schema.db_id, item.orig['query'], code)
                beam_dict[i] = ret_dict
                if ret_dict["exact"] is True:
                    break
            self.results.append(beam_dict)

        def finalize(self):
            self.evaluator.finalize()
            return {
                'per_item': self.results,
                'total_scores': self.evaluator.scores
            }


class SparcDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="sparc/", batch_size=4, tokenizer=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        assert tokenizer is not None
        self.dataset = {}

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            for split in ('train', 'dev'):
                dataset = SparcDataset(os.path.join(self.data_dir, f'{split}.json'),
                                       os.path.join(self.data_dir, 'tables.json'),
                                       os.path.join(self.data_dir, 'database'),
                                       tokenizer=self.tokenizer, mode=split)
                self.dataset[split] = dataset
        if stage == 'test' or stage is None:
            test_dataset = SparcDataset(os.path.join(self.data_dir, 'dev.json'),
                                        os.path.join(self.data_dir, 'tables.json'),
                                        os.path.join(self.data_dir, 'database'),
                                        tokenizer=self.tokenizer, mode='test')
            self.dataset['test'] = test_dataset

    def train_dataloader(self):
        if 'train' not in self.dataset:
            dataset = SparcDataset(os.path.join(self.data_dir, 'train.json'),
                                   os.path.join(self.data_dir, 'tables.json'),
                                   os.path.join(self.data_dir, 'database'),
                                   tokenizer=self.tokenizer, mode='train')
            self.dataset['train'] = dataset
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, collate_fn=SparcDataset.collate_fn)

    def val_dataloader(self):
        if 'dev' not in self.dataset:
            dataset = SparcDataset(os.path.join(self.data_dir, 'dev.json'),
                                   os.path.join(self.data_dir, 'tables.json'),
                                   os.path.join(self.data_dir, 'database'),
                                   tokenizer=self.tokenizer, mode='dev')
            self.dataset['dev'] = dataset
        return DataLoader(self.dataset['dev'], batch_size=self.batch_size, collate_fn=SparcDataset.collate_fn)

    def test_dataloader(self):
        if 'test' not in self.dataset:
            dataset = SparcDataset(os.path.join(self.data_dir, 'dev.json'),
                                   os.path.join(self.data_dir, 'tables.json'),
                                   os.path.join(self.data_dir, 'database'),
                                   tokenizer=self.tokenizer, mode='test')
            self.dataset['test'] = dataset
        return DataLoader(self.dataset['test'], batch_size=self.batch_size, collate_fn=SparcDataset.collate_fn)


if __name__ == '__main__':
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', additional_special_tokens=['<c>', '<space>'])
    # train_data = SparcDataset('sparc/train.json', 'sparc/tables.json', 'sparc/database', bart_tokenizer)
    # dataloader = torch.utils.data.DataLoader(train_data, batch_size=7, collate_fn=train_data.collate_fn)
    sparc_data = SparcDataModule('sparc/', batch_size=7)
    for batch in sparc_data.train_dataloader():
        a = 1