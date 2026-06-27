import os
import argparse
import json
import re
import shutil
from openai import OpenAI
from multiprocessing import Pool, cpu_count
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from collections import Counter
from collections import defaultdict

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default='./playground/Instructions_slim/VizWiz/val_new.json')
    parser.add_argument('--result-file', type=str, default='./results/CoIN_slim_new/VizWiz/Zero_shot/merge.jsonl')
    parser.add_argument('--output-dir', type=str)
    return parser.parse_args()

def create_coco_type(annotation_file, result_file, output_dir):
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    total = len(results)
    right = 0
    coco_results = []
    image_id = 1
    for result in results:
        pred = result['text']

        coco_results.append({
            "image_id": int(image_id),  # 确保 image_id 是整数类型
            "caption": pred
        })
        image_id += 1
    output_file = 'pred_coco_type.json'
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w') as f_out:
        json.dump(coco_results, f_out, indent=4)
    return output_path, total

def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_captions(pred_file, val_file, output_file):
    # 加载预测文件和验证文件
    pred_data = load_json(pred_file)
    val_data = load_json(val_file)

    # 将验证数据中的 captions 根据 image_id 组织成字典，image_id -> [captions]
    val_dict = defaultdict(list)
    for item in val_data['annotations']:
        val_dict[item['image_id']].append(item['caption'])

    # 合并预测数据与验证数据
    merged_data = []
    for pred_item in pred_data:
        image_id = pred_item['image_id']
        pred_caption = pred_item['caption']

        # 获取真实值 captions
        gt_captions = val_dict.get(image_id, [])

        # 将预测与真实值合并
        merged_data.append({
            "pred": pred_caption,
            "ground_truth": gt_captions
        })

    # 将合并结果保存为 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

def load_coco(annotation_file):
    with open(annotation_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    anns = dataset.get("annotations", [])
    has_missing_category_id = any(isinstance(ann, dict) and "category_id" not in ann for ann in anns)
    if "categories" in dataset and has_missing_category_id:
        dataset = dict(dataset)
        dataset.pop("categories", None)

    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    return coco

def simple_tokenize(text):
    if text is None:
        return []
    return re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", str(text).lower())

def lcs_length(a, b):
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, start=1):
            tmp = dp[j]
            if x == y:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]

def rouge_l_f1(hyp_tokens, ref_tokens, beta=1.2):
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(hyp_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    prec = lcs / len(hyp_tokens)
    rec = lcs / len(ref_tokens)
    denom = rec + (beta * beta) * prec
    if denom == 0:
        return 0.0
    return ((1 + beta * beta) * prec * rec) / denom

def safe_meteor(references, hypothesis):
    try:
        return float(meteor_score(references, hypothesis))
    except Exception:
        return 0.0

def eval_with_python_metrics(coco, output_file, total):
    preds = load_json(output_file)

    references = []
    hypotheses = []
    meteor_scores = []
    rouge_scores = []

    for item in preds:
        image_id = item.get("image_id")
        hyp = item.get("caption", "")
        anns = coco.imgToAnns.get(image_id, [])
        refs = [ann.get("caption", "") for ann in anns if isinstance(ann, dict)]
        if not refs:
            continue

        references.append([simple_tokenize(r) for r in refs])
        hypotheses.append(simple_tokenize(hyp))
        meteor_scores.append(safe_meteor(refs, hyp))
        hyp_toks = simple_tokenize(hyp)
        rouge_scores.append(max((rouge_l_f1(hyp_toks, simple_tokenize(r)) for r in refs), default=0.0))

    if not hypotheses:
        bleu_1 = bleu_2 = bleu_3 = bleu_4 = 0.0
        meteor = 0.0
        rouge_l = 0.0
    else:
        smoother = SmoothingFunction().method1
        bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoother)
        bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
        bleu_3 = corpus_bleu(references, hypotheses, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smoother)
        bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)
        meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
        rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    cider = 0.0
    return {
        "Bleu_1": bleu_1,
        "Bleu_2": bleu_2,
        "Bleu_3": bleu_3,
        "Bleu_4": bleu_4,
        "METEOR": meteor,
        "ROUGE_L": rouge_l,
        "CIDEr": cider,
    }

def eval_single(output_file, annotation_file, total):
    coco = load_coco(annotation_file)  # Ground truth JSON file
    coco_res = coco.loadRes(output_file)  # Prediction JSON file

    metrics_to_print = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]
    eval_dict = None

    if shutil.which("java") is None:
        eval_dict = eval_with_python_metrics(coco, output_file, total)
    else:
        try:
            coco_eval = COCOEvalCap(coco, coco_res)
            coco_eval.evaluate()
            eval_dict = coco_eval.eval
        except FileNotFoundError:
            eval_dict = eval_with_python_metrics(coco, output_file, total)

    results = []
    for metric in metrics_to_print:
        score = float(eval_dict.get(metric, 0.0))
        score_percentage = score * 100.0
        print(f"{metric}: {score_percentage:.2f}")
        results.append(score_percentage)

    avg = sum(results) / len(results) if results else 0.0
    print('Samples: {}\nAverage: {:.2f}%\n'.format(total, avg))
    #将结果写入文件
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nBleu_1: {:.2f}\nBleu_2: {:.2f}\nBleu_3: {:.2f}\nBleu_4: {:.2f}\nMETEOR: {:.2f}\nROUGE_L: {:.2f}\nCIDEr: {:.2f}\nAverage: {:.2f}\n'.format(
                total, results[0], results[1], results[2], results[3], results[4], results[5], results[6], avg))
    

def process_batch(api_key, batch):
    """
    对一个批次的数据进行评分，返回该批次所有样本的评分列表。
    """
    from openai import OpenAI  # 确保每个子进程加载必要的模块

    client = OpenAI(api_key=api_key, base_url="https://platform.llmprovider.ai/v1")

    message = (
        "Below are the model's predictions and the ground truth answers for a task. "
        "For each case, provide a semantic similarity score between 0 and 10 in the format 'Score: X', "
        "where X is your score. Always use the format 'Score:' and do not explain anything."
        "\n\nResults:\n" +
        "\n".join([f"{i+1}. Pred: {item['pred']}, Ground Truth: {item['ground_truth']}" for i, item in enumerate(batch)])
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an AI assistant evaluating a model's prediction quality"}, {"role": "user", "content": message}],
        stream=False
    )

    evaluation_text = response.choices[0].message.content

    # 提取评分
    scores = []
    for line in evaluation_text.splitlines():
        score = float(line.split(":")[1].strip())
        scores.append(score)
    return scores

def deepseek_chat_final(api_key, path, batch_size=10):
    """
    使用多进程评估，返回所有样本的最终平均准确率。
    """
    # 加载 JSON 文件
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 分批处理
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    num_batches = len(batches)

    print(f"Total data: {len(data)}, Total batches: {num_batches}, Batch size: {batch_size}")

    # 使用多进程池处理所有批次
    total_score = 0
    total_samples = 0
    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            process_batch, [(api_key, batch) for batch in batches]
        )

        # 累计每个批次的评分总和和样本数
        for batch_scores in results:
            total_score += sum(batch_scores)
            total_samples += len(batch_scores)

    # 计算总体评分平均值
    overall_average_score = total_score / total_samples if total_samples > 0 else 0
    return overall_average_score
    



if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        output_file, total = create_coco_type(args.annotation_file, args.result_file, args.output_dir)
        ans_gt_file = os.path.join(args.output_dir, 'ans_gt.json')
        merge_captions(output_file, args.annotation_file, ans_gt_file)
        eval_single(output_file, args.annotation_file, total)

        # api_key = "sk-GdmqmU6fFWv5N0HlvYluLFzIbXIPNg3MHzPGeeV247092807Ba2e4487B9D5796cA3Be7dD4"

        # batch_size = 8 
        # overall_accuracy = deepseek_chat_final(api_key, ans_gt_file, batch_size=batch_size)
        # print(f"Overall Accuracy: {overall_accuracy*10:.2f}")
        # if args.output_dir is not None:
        #     output_file = os.path.join(args.output_dir, 'Result_api.text')
        #     with open(output_file, 'w') as f:
        #         f.write('Accuracy: {:.2f}%\n'.format(overall_accuracy*10))
