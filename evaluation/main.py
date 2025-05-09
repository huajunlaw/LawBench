import json
import os
import pandas as pd
from evaluation_functions import jec_ac, jec_kd, cjft, ydlj, ftcs, jdzy, jetq, ljp_accusation, ljp_article, ljp_imprison, wbfl, xxcq, flzx, wsjd, yqzy, lblj, zxfl, sjjc
import sys
import argparse

def read_json(input_file):
    # load the json file
    with open(input_file, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    return list(data_dict.values())
task_dict = {
  "1-1": {
    "task_name": "法条背诵",
    "data_source": "FLK",
    "metrices": "ROUGE-L",
    "type": "生成"
  },
  "1-2": {
    "task_name": "知识问答",
    "data_source": "JEC_QA",
    "metrices": "Accuracy",
    "type": "单选"
  },
  "2-1": {
    "task_name": "文件校对",
    "data_source": "CAIL2022",
    "metrices": "F0.5",
    "type": "生成"
  },
  "2-2": {
    "task_name": "纠纷焦点识别",
    "data_source": "LAIC2021",
    "metrices": "F1",
    "type": "多选"
  },
  "2-3": {
    "task_name": "婚姻纠纷鉴定",
    "data_source": "AIStudio",
    "metrices": "F1",
    "type": "多选"
  },
  "2-4": {
    "task_name": "问题主题识别",
    "data_source": "CrimeKgAssitant",
    "metrices": "Accuracy",
    "type": "单选"
  },
  "2-5": {
    "task_name": "阅读理解",
    "data_source": "CAIL2019",
    "metrices": "rc-F1",
    "type": "抽取"
  },
  "2-6": {
    "task_name": "命名实体识别",
    "data_source": "CAIL2021",
    "metrices": "soft-F1",
    "type": "抽取"
  },
  "2-7": {
    "task_name": "舆情摘要",
    "data_source": "CAIL2022",
    "metrices": "ROUGE-L",
    "type": "生成"
  },
  "2-8": {
    "task_name": "论点挖掘",
    "data_source": "CAIL2022",
    "metrices": "Accuracy",
    "type": "单选"
  },
  "2-9": {
    "task_name": "事件检测",
    "data_source": "LEVEN",
    "metrices": "F1",
    "type": "多选"
  },
  "2-10": {
    "task_name": "触发词提取",
    "data_source": "LEVEN",
    "metrices": "soft-F1",
    "type": "抽取"
  },
  "3-1": {
    "task_name": "法条预测(基于事实)",
    "data_source": "CAIL2018",
    "metrices": "F1",
    "type": "多选"
  },
  "3-2": {
    "task_name": "法条预测(基于场景)",
    "data_source": "LawGPT_zh Project",
    "metrices": "ROUGE-L",
    "type": "生成"
  },
  "3-3": {
    "task_name": "罪名预测",
    "data_source": "CAIL2018",
    "metrices": "F1",
    "type": "多选"
  },
  "3-4": {
    "task_name": "刑期预测(无法条内容)",
    "data_source": "CAIL2018",
    "metrices": "Normalized log-distance",
    "type": "回归"
  },
  "3-5": {
    "task_name": "刑期预测(给定法条内容)",
    "data_source": "CAIL2018",
    "metrices": "Normalized log-distance",
    "type": "回归"
  },
  "3-6": {
    "task_name": "案例分析",
    "data_source": "JEC_QA",
    "metrices": "Accuracy",
    "type": "单选"
  },
  "3-7": {
    "task_name": "犯罪金额计算",
    "data_source": "LAIC2021",
    "metrices": "Accuracy",
    "type": "回归"
  },
  "3-8": {
    "task_name": "咨询",
    "data_source": "hualv.com",
    "metrices": "ROUGE-L",
    "type": "生成"
  }
}
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", dest="input_folder",
                  help="input folder: it should be a folder containing the prediction results", metavar="FILE")
    parser.add_argument("-o", "--outfile", dest="outfile",
                  help="output file saving the evaluation results", metavar="FILE")
    args = parser.parse_args(argv)
    funct_dict = {"3-6": jec_ac.compute_jec_ac,
                  "1-2": jec_kd.compute_jec_kd,
                  "3-2": cjft.compute_cjft,
                  "3-8": flzx.compute_flzx,
                  "1-1": ftcs.compute_ftcs,
                  "2-2": jdzy.compute_jdzy,
                  "3-7": jetq.compute_jetq,
                  "3-3": ljp_accusation.compute_ljp_accusation,
                  "3-1": ljp_article.compute_ljp_article,
                  "3-4": ljp_imprison.compute_ljp_imprison,
                  "3-5": ljp_imprison.compute_ljp_imprison,
                  "2-3": wbfl.compute_wbfl,
                  "2-6": xxcq.compute_xxcq,
                  "2-1": wsjd.compute_wsjd,
                  "2-4": zxfl.compute_zxfl,
                  "2-7": yqzy.compute_yqzy,
                  "2-8": lblj.compute_lblj,
                  "2-5": ydlj.compute_ydlj,
                  "2-9": sjjc.compute_sjjc,
                  "2-10": sjjc.compute_cfcy}
    input_dir = args.input_folder

    output_file = args.outfile
    results = {"task": [], "model_name": [], "score": [], "abstention_rate": [], "task_type": []}
    # list all folders in input_dir
    system_folders = os.listdir(input_dir)
    for system_folder in system_folders:
        if system_folder.startswith("."):
            continue
        if system_folder.startswith("llama"):
            continue
        if system_folder.endswith("hf"):
            continue
        if system_folder in ('GPT-3.5-turbo-0613', 'GPT4', 'gogpt-7b',):
            continue
        system_folder_dir = os.path.join(input_dir, system_folder)
        if not os.path.isdir(system_folder_dir):
            continue
        # list all files in system_folder_path
        dataset_files = os.listdir(system_folder_dir)
        print(f"*** Evaluating System: {system_folder} ***")
        for dataset_file in dataset_files:
            datafile_name = dataset_file.split(".")[0]
            input_file = os.path.join(system_folder_dir, dataset_file)
            data_dict = read_json(input_file)
            if datafile_name not in funct_dict:
                print(f"*** Warning: {datafile_name} is not in funct_dict ***")
                continue
            print(f"Processing {datafile_name}:")
            score_function = funct_dict[datafile_name]
            score = score_function(data_dict)
            print(f"Score of {datafile_name}: {score}")
            results["task"].append(task_dict[datafile_name]['task_name'])
            results["metrics"].append(task_dict[datafile_name]['metrices'])
            results["model_name"].append(system_folder)
            results["score"].append(score["score"])
            abstention_rate = score["abstention_rate"] if "abstention_rate" in score else 0
            results["abstention_rate"].append(abstention_rate)

        print(f"*** Evaluating System: {system_folder} Done! ***")
        print()
        print()

    results = pd.DataFrame(results)
    results.to_csv(output_file, index = False)

if __name__ == '__main__':
    main(sys.argv[1:])
