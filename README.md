# Large Language Model Distilling Medication Recommendation Model

This is the official implementation of the paper "Large Language Model Distilling Medication Recommendation Model".

## Running

You can implement our model according to the following steps:

1. Prepare the LlaMA-7B model: download all files of LlaMA-7B and put them into `resources/llama-7b/`.
2. Prepare the data: apply the data from the [officail website](https://mimic.mit.edu/), and put the unzipped raw data into `data/mimic3/raw/` and `data/mimic4/`, respectively. Then, run the scripts `construction.ipynb` under `data/mimic3/` and `data/mimic4/` to preprocess the data. The preprocessed data will be saved under `mimic3/handled/` and `mimic4/handled/.` Besides, the file to convert ATC code to drug name is available from this [link](https://github.com/fabkury/atcd), i.e., "WHO ATC-DDD 2021-12-03.csv". Other auxiliay files, such as "drug-DDI.csv" can be otained from the repo of [GAMENet](https://github.com/sjy1203/GAMENet) and [SafeDrug](https://github.com/ycq091044/SafeDrug).
3. Install the necessary packages. Run the command:

   ```bash
   pip install -r requirements.txt
   ```
4. First, train the large language model for medication recommendation via the command:

   ```bash
   bash experiments/llm_cls.bash
   ```
5. Then, you can run the knowledge distillation via the following command:

   ```bash
   bash experiments/mimic3/online_distill.bash
   bash experiments/mimic4/online_distill.bash
   ```
6. For the long running time of distillation, we can save the hidden states from LLM previously. You can run the test on the train file, and the hidden states will be saved in the results automatically vias our `llm_cls.bash`. Then, put the results file into `mimic3/handled/` or `mimic4/handled/`, then run the KD within two hours!

   ```bash
   bash experiments/mimic3/offline_distill.bash
   bash experiments/mimic4/offline_distill.bash
   ```

## Citation

If the code and the paper are useful for you, it is appreciable to cite our paper:

```
@article{liu2024large,
  title={Large Language Model Distilling Medication Recommendation Model},
  author={Liu, Qidong and Wu, Xian and Zhao, Xiangyu and Zhu, Yuanshao and Zhang, Zijian and Tian, Feng and Zheng, Yefeng},
  journal={arXiv preprint arXiv:2402.02803},
  year={2024}
}
```

## Thanks

The code refers to the repo [MOELoRA-peft](https://github.com/liuqidong07/MOELoRA-peft), [GAMENet](https://github.com/sjy1203/GAMENet) and [SafeDrug](https://github.com/ycq091044/SafeDrug).
