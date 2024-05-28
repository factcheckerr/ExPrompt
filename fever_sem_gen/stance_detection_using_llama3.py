import json

import requests
import subprocess

from nltk.tokenize import sent_tokenize
def tokenize_into_sentences(paragraph):
    sentences = sent_tokenize(paragraph)
    return sentences





# llama3:70b
# mixtral:8x7b
class LLMStanceDetector:
    def __init__(self, model: str = "llama3:70b",
                 url: str = "http://tentris-ml.cs.upb.de:8000/api/generate"):
        self.model = model
        self.url = url

    def get_response_from_api_call(self, text: str, claim: str):
        """
        :param text: String representation of an OWL Class Expression
        """
        # prompt=(f"<s> [INST] You are an expert in stance detection. "
        #         f"You have only 3 options (CONTRADICTS, SUPPORT, and NOT ENOUGH INFO) to detect the stance from the following textual description representing the given claim."
        #         f"[/INST] Model answer</s> [INST] "
        #         f"Given claim: {claim}."
        #         f"Textual description: {text}."
        #         f"Answer only a single option from the three: (1) CONTRADICTS, (2) SUPPORTS, and (3) NOT ENOUGH INFO. "
        #         f"Provide no explanations or write no notes.[/INST]")


        example1 = (f"Given claim is: Saturn is the largest planet in the Solar System. "
                   f"Textual document is: Saturn is the sixth planet from the Sun and the second-largest in the Solar System , after Jupiter. ")
        example2 = (f"Given claim is: The Pelican Brief is based solely on a television series. "
                   f"Textual document is:  The Pelican Brief is a 1993 American legal political thriller based on the television series of the same name by John Grisham .")

        # example = example.replace("\n", "")
        prompt = (f"<s> [INST] You are an expert in stance detection. "
                  f"You only have three options (REFUTES and SUPPORTS) to detect stance from a textual document: refuting or supporting the given claim."
                  f"Only output must be one of these three options: (1) REFUTES or (2) SUPPORTS. \n"
                  f"Examples are the following:"
                  f"Example 1: {example1}\n[/INST]"
                  f"Answer: REFUTES."    
                  f"[INST]Example 2: {example2}\n[/INST]"
                  f"Answer: SUPPORTS."
                  f"[INST]You should not output more than one word, i.e., (1) REFUTES or (2) SUPPORTS. "
                  f"Do not output true or false, rather (1) REFUTES or (2) SUPPORTS, for refuting or supporting for the given claim.  "
                  f"The only output must be one of these three options: (1) REFUTES or (2) SUPPORTS."
                  f"The output should not contain explanations, notes, or numbers, and it should not begin with a number. [/INST]</s>\n"
                  f"[INST]The given claim is: {claim}\n Given textual document is: {text} [/INST] ")

        # prompt = prompt.replace("\n","")

        # print(prompt)
        response = requests.get(url=self.url,
                                headers={"accept": "application/json", "Content-Type": "application/json"},
                                json={"model": self.model, "prompt": prompt})
        # print(response.json()["response"])
        return response.json()["response"]



    # def get_triples_from_transformer(self, text: str, claim: str):
    #     """
    #     :param text: String representation of an OWL Class Expression
    #     """
    #
    #     prompt=(f"<s> [INST] You are an expert in linguistics and stance detection. "
    #                 f"Given claim: {claim}."
    #                 f"You have only 3 options (REFUTES, SUPPORTS, and NOT ENOUGH INFO) to detect the stance from the following textual description representing the given claim."
    #                 f"Textual description: {text}."
    #                 f"[/INST] Model answer</s> [INST] "
    #                 f"Answer only a single option from the three: (1) REFUTES, (2) SUPPORTS, and (3) NOT ENOUGH INFO. "
    #                 f"Provide no explanations or write no notes.[/INST]")
    #
    #     # print(prompt)
    #     print("--------------------------------------------------------------LLM Starts here!!!--------------------------------------------")
    #
    #     response = pipe(prompt)[0]['generated_text']
    #
    #     response = response.split("[/INST]")[-1]
    #
    #     # response = response.split(" (")[1]
    #     return response


verbalizer = LLMStanceDetector()
#
#
#
claim_lines = []
# Read data from a file (assuming it's in JSON format)
count = 0
typ = 'full'
model = 'llama3'
data2 = model+'_llm_output_'+typ+'.jsonl'
input_file = 'fever_symmetric_'+typ+'.jsonl'
#
path = '/local/upb/users/u/uqudus/profiles/unix/cs/fever/fever_symmetric/v-0_1/'
# file_path = path + 'corpus.jsonl'
with open(path + data2, 'r') as output_file:
    for line in output_file:
        # print(line)
        cdata = json.loads(line)
        cc = cdata['claim']
        claim_lines.append(cc)

#
with open(path+input_file, 'r') as f, open(path+data2, 'a') as output_file:
    for line in f:
        data = json.loads(line)
        claim = data['claim']
        if claim not in claim_lines:
            claim_lines.append(claim)
            print(str(data).replace("\n",""))
            ev2 = data['evidence_sentence']
            print(str(claim))
            claim_lines.append(claim)
            # line4 = verbalizer.get_line_from_file(file_path, ev2)
            # title = json.loads(line4)['title']
            # abstract = ' '.join(json.loads(line4)['abstract'])
            # text = 'title: '+title + ' content: ' + abstract

            print("claim:" + claim)
            # print("ground truth label:" + data['label'])
            # sentences = tokenize_into_sentences(ev2.replace("\n", " "))
            # print(" ".join(sentences[:5]))
            # exit(1)
            # print(len(sentences))
            # text = " ".join(sentences[:500])
            answer = verbalizer.get_response_from_api_call(text=ev2, claim=claim)
            answer = answer.replace("\n", " ")
            # print("answer is:"+ answer)
            final_verdict = "NOT ENOUGH INFO"
            if "Answer: REFUTES" in answer or ("no evidence that support the claim".lower() in answer.lower() or (
                    "REFUTES".lower() in answer.lower() and "SUPPORTS".lower() not in answer.lower())):
                final_verdict = "REFUTES"
                print("ANSWER->REFUTES")
            elif "Answer: SUPPORTS" in answer or (
                    "SUPPORTS".lower() in answer.lower() and "REFUTES".lower() not in answer.lower()):
                final_verdict = "SUPPORTS"
                print("ANSWER->SUPPORTS")
            elif "NOT ENOUGH INFO".lower() in answer.lower():
                print("ANSWER->NOT ENOUGH INFO")
                final_verdict = "NOT ENOUGH INFO"
            else:
                print(ev2)
                print("answer is: " + answer)
                # break
            #
            data['LLM_prediction'] = final_verdict
            # Write the updated JSON object to the output file
            json.dump(data, output_file)
            output_file.write('\n')
            #
            #             # print(data)
            print("--------------------------------claim ENDS---------------------------")
            count = count + 1
            # if count >1000:
            #     break
# print(lines)
# exit(1)

