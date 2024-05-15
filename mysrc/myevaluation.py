import subprocess
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu


def run_script_of_pass_k(problems_jsonl, samples_jsonl):
    result = subprocess.run(["evaluate_functional_correctness", samples_jsonl, "--problem_file=" + problems_jsonl],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            encoding="utf-8")
    if result.returncode == 0:
        print("success:", result)
    else:
        print("error:", result)


def run_script_of_bleu(reference, candidate):
    score = sentence_bleu(reference, candidate)
    return score


def run_script_of_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    results = scorer.score(target=reference, prediction=candidate)
    return results
