import sys
import os
import dspy
from dspy.datasets import HotPotQA
import dspy.evaluate
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate
from dsp.utils import deduplicate
from rich import print

# 1. Configuration & Data Loading
turbo = dspy.OpenAI(model="gpt-3.5-turbo")
colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)
dataset = HotPotQA(train_seed=1, train_size=20,
                   eval_seed=2023, dev_size=50, test_size=0)

trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

print(f"Trainset Data {trainset[:5]}")
print(f"Devset Data {devset[:5]}")

print("\n## Example Question with Answer ###\n")
example = devset[18]
print(f"Question: {example.question}")
print(f"Answer: {example.answer}")
print(f"Relevant Wikipedia Titles: {example.gold_titles}")


# 2. Basic Chatbot
class BasicQA(dspy.Signature):  # Signature
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


print("\n### Generate Response $$$\n")
generate_answer = dspy.Predict(BasicQA)
pred = generate_answer(question=example.question)
print(f"Question: {example.question}\nPredicted Answer: {pred.answer}")


# 3. Chatbot with Chain of Thought
print("\n### Generate Response with Chain of Thought ###\n")
generate_answer_with_chain_of_thought = dspy.ChainOfThought(BasicQA)
pred = generate_answer_with_chain_of_thought(question=example.question)
print(
    f"Question: {example.question}\nThought: {pred.rationale.split('.', 1)[1].strip()}\nPredicted Answer: {pred.answer}")


# 4. Chatbot with Chain of Thought and Context = RAG -> (Retrieve, Generate Response)
print("\n### RAG: Generate Response with Chain of Thought and Context ###\n")

# 4a. Signature


class GenerateAnswer(dspy.Signature):  # Signature
    """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# 4b. Module / Pipeline


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


# 4c. Optimizer / Optimizing Pipeline
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM, answer_PM


teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

my_question = "What year was manchester united formed?"
pred = compiled_rag(my_question)

print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")
print(
    f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")


# # 5. Evaluating the Answers
# print("\n### Evaluating the Answers ###\n")

# # 5a. Basic RAG


# def gold_passages_retrieved(example, pred, trace=None):
#     gold_titles = set(
#         map(dspy.evaluate.normalize_text, example['gold_titles']))
#     found_titles = set(map(dspy.evaluate.normalize_text, [
#                        c.split(' | ')[0] for c in pred.context]))
#     return gold_titles.issubset(found_titles)


# evaluate_on_hotpotqa = Evaluate(
#     devset=devset, num_threads=1, display_progress=True, display_table=5)
# compiled_rag_retrieval_score = evaluate_on_hotpotqa(
#     compiled_rag, metric=gold_passages_retrieved)
