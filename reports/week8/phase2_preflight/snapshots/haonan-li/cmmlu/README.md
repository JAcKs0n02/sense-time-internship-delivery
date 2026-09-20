---
license: cc-by-nc-4.0
task_categories:
- multiple-choice
- question-answering
language:
- zh
tags:
- chinese
- llm
- evaluation
pretty_name: CMMLU
size_categories:
- 10K<n<100K
---

# CMMLU: Measuring massive multitask language understanding in Chinese

- **Homepage:** [https://github.com/haonan-li/CMMLU](https://github.com/haonan-li/CMMLU)
- **Repository:** [https://huggingface.co/datasets/haonan-li/cmmlu](https://huggingface.co/datasets/haonan-li/cmmlu)
- **Paper:** [CMMLU: Measuring Chinese Massive Multitask Language Understanding](https://arxiv.org/abs/2306.09212).



## Table of Contents

- [Introduction](#introduction)
- [Leaderboard](#leaderboard)
- [Data](#data)
- [Citation](#citation)
- [License](#license)

## Introduction

CMMLU is a comprehensive Chinese assessment suite specifically designed to evaluate the advanced knowledge and reasoning abilities of LLMs within the Chinese language and cultural context. 
CMMLU covers a wide range of subjects, comprising 67 topics that span from elementary to advanced professional levels. It includes subjects that require computational expertise, such as physics and mathematics, as well as disciplines within humanities and social sciences. 
Many of these tasks are not easily translatable from other languages due to their specific contextual nuances and wording. 
Furthermore, numerous tasks within CMMLU have answers that are specific to China and may not be universally applicable or considered correct in other regions or languages.

## Leaderboard

Latest leaderboard is in our [github](https://github.com/haonan-li/CMMLU).

## Data 

We provide development and test dataset for each of 67 subjects, with 5 questions in development set and 100+ quesitons in test set.

Each question in the dataset is a multiple-choice questions with 4 choices and only one choice as the correct answer. 

Here are two examples:
```
    题目：同一物种的两类细胞各产生一种分泌蛋白，组成这两种蛋白质的各种氨基酸含量相同，但排列顺序不同。其原因是参与这两种蛋白质合成的：
    A. tRNA种类不同
    B. 同一密码子所决定的氨基酸不同
    C. mRNA碱基序列不同
    D. 核糖体成分不同
    答案是：C

```

```
    题目：某种植物病毒V是通过稻飞虱吸食水稻汁液在水稻间传播的。稻田中青蛙数量的增加可减少该病毒在水稻间的传播。下列叙述正确的是：
    A. 青蛙与稻飞虱是捕食关系
    B. 水稻和病毒V是互利共生关系
    C. 病毒V与青蛙是寄生关系
    D. 水稻与青蛙是竞争关系
    答案是： 
```

#### Load data

```python
from datasets import load_dataset
cmmlu=load_dataset(r"haonan-li/cmmlu", 'agronomy')
print(cmmlu['test'][0])
```
#### Load all data at once
```python
task_list = ['agronomy', 'anatomy', 'ancient_chinese', 'arts', 'astronomy', 'business_ethics', 'chinese_civil_service_exam', 'chinese_driving_rule', 'chinese_food_culture', 'chinese_foreign_policy', 'chinese_history', 'chinese_literature', 
'chinese_teacher_qualification', 'clinical_knowledge', 'college_actuarial_science', 'college_education', 'college_engineering_hydrology', 'college_law', 'college_mathematics', 'college_medical_statistics', 'college_medicine', 'computer_science',
'computer_security', 'conceptual_physics', 'construction_project_management', 'economics', 'education', 'electrical_engineering', 'elementary_chinese', 'elementary_commonsense', 'elementary_information_and_technology', 'elementary_mathematics', 
'ethnology', 'food_science', 'genetics', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_geography', 'high_school_mathematics', 'high_school_physics', 'high_school_politics', 'human_sexuality',
'international_law', 'journalism', 'jurisprudence', 'legal_and_moral_basis', 'logical', 'machine_learning', 'management', 'marketing', 'marxist_theory', 'modern_chinese', 'nutrition', 'philosophy', 'professional_accounting', 'professional_law', 
'professional_medicine', 'professional_psychology', 'public_relations', 'security_study', 'sociology', 'sports_science', 'traditional_chinese_medicine', 'virology', 'world_history', 'world_religions']

from datasets import load_dataset
cmmlu = {k: load_dataset(r"haonan-li/cmmlu", k) for k in task_list}

```


## Citation
```
@misc{li2023cmmlu,
      title={CMMLU: Measuring massive multitask language understanding in Chinese}, 
      author={Haonan Li and Yixuan Zhang and Fajri Koto and Yifei Yang and Hai Zhao and Yeyun Gong and Nan Duan and Timothy Baldwin},
      year={2023},
      eprint={2306.09212},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

The CMMLU dataset is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

