#%%
import numpy as np
from typing import List, Dict
from nlgmetricverse import load_metric, NLGMetricverse
from nlgmetricverse.metrics import Bertscore



class NLGEvaluator:
    def __init__(self, device='cpu'):
        # Initialize ROUGE, CIDEr, and BERT scorers
        self.nlg_metricverse = NLGMetricverse(metrics=[
            load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
            load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
            # load_metric("bertscore", compute_kwargs={'device': device, 'idf': True}),
            # load_metric('meteor'),
            load_metric("rouge"),
            load_metric("cider"),
            # load_metric('recall')
        ])

    def evaluate(self, pred_sentences: List[str], reference_sentences: List[List[str]]):
        metricverse_results = self.nlg_metricverse(predictions=pred_sentences, references=reference_sentences)
        log_dict = {
            'nlg/blue_1': metricverse_results['bleu_1']['score'],
            'nlg/blue_4': metricverse_results['bleu_4']['score'],
            'nlg/rouge_1': metricverse_results['rouge']['rouge1'],
            'nlg/rouge_2': metricverse_results['rouge']['rouge2'],
            'nlg/rouge_L': metricverse_results['rouge']['rougeL'],
            'nlg/cider': metricverse_results['cider']['score'],
            # 'nlg/bertscore': metricverse_results['bertscore']['score'],
            # 'nlg/bertscore_p': np.mean(metricverse_results['bertscore']['precision']),
            # 'nlg/bertscore_r': np.mean(metricverse_results['bertscore']['recall']),
            # 'nlg/bertscore_f1': np.mean(metricverse_results['bertscore']['f1']),
            # 'nlg/meteor': metricverse_results['meteor']['score'],
            'nlg/recall': metricverse_results['recall']['score']
        }
        log_dict['nlg/scores_sum'] = sum(log_dict.values())
        return log_dict


#%%
if __name__ == '__main__':
    m = NLGEvaluator()
    pred = ['a person waves left hand', 'a person is dancing Waltz', 'i love python', 'world peace'] * 8
    src = [['the human is reaching out his left hand',  'hello world']] * 32
    res = m.evaluate(pred, src)

    for k, v in res.items():
        print(f'{k}: {v}')
# %%
