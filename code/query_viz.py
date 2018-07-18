import json
import os

import matplotlib.pyplot as plt

import data_pull as dp
import image_fetch_wrappers as ifw

with open('config.json') as f:
    cfg_data = json.load(f)
    out_path = cfg_data['file_paths']['output_path']


class ImageQueryData(object):
    def __init__(query, query_id, image_id, if_data):
        self.query = query
        self.query_id = query_id
        self.image_id = image_id
        self.if_data = if_data
        self.query_text = self.get_query_text()

    def get_query_text(self):
        import pdb; pdb.set_trace()

    def run_crf(self):
        data = ifw.inference_pass(self.query, self.query_id, self.image_id,
                                  self.if_data, 'original')
        _, _, energy, best_matches, _, _ = data
        return energy, best_matches

    def run_unary_models(self):
        pass

    def run_binary_models(self):
        pass

    def compute_data(self):
        self.energy, self.best_matches = self.run_crf()
        self.run_unary_models()
        self.run_binary_models()

    def generate_plot(self, output_path=None):
        pass


if __name__ == '__main__':
    _, _, _, _, queries, if_data = dp.get_all_data(use_csv=True)
    import pdb; pdb.set_trace()
    iqd = ImageQueryData(queries[0], if_data)
    print(iqd.query_text)
    # iqd.compute_data()
    # iqd.generate_plot()

