# Code implementing ahmad's path on text aware visual features
## Instructions to replicate



1. Install Packages

```Shell
conda create -n tcv python=3.10 -y
conda activate tcv
pip install --upgrade pip  
pip install -r requirements/tcv_requirements.txt
```

2. Download Coco Training data and unzip the instructions file

```Shell

unzip ./modules/ahmad_conditional_visual/train-data-shard/coco_only_364k_new.json.zip -d ./modules/ahmad_conditional_visual/train-data-shard/coco_only_364k_new.json

wget -P modules/ahmad_conditional_visual/train-data-shard/train2017.zip http://images.cocodataset.org/zips/train2017.zip 

unzip modules/ahmad_conditional_visual/train-data-shard/train2017.zip 

```

3. Run Training Script
```Shell

sh ./scripts/tcv/train_tcv.sh

```

4. Run Evaluation Script
```Shell

sh ./scripts/tcv/eval_mmvet.sh

```

5. Submit results `./modules/ahmad_conditional_visual/data/eval/mmvet/mm-vet/results/TCV_phi3.jsonl` to this [MM-Vet Evaluator](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator)



Mine and LLava's Results are available at `./modules/ahmad_conditional_visual/eval/mmvet/results/`
My results are under the following wild card `./modules/ahmad_conditional_visual/eval/mmvet/results/TCV_phi3*`