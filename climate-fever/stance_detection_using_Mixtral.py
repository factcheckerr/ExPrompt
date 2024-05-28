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
    def __init__(self, model: str = "mixtral:8x7b",
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

        example1 = (f"Given claim is: The polar bear population has been growing. "
                   f"Sentences refuting the given claim from the given textual document are: In two areas where harvest levels have been increased based on increased sightings, science-based studies have indicated declining populations, and a third area is considered data-deficient. Of the 19 recognized polar bear subpopulations, one is in decline, two are increasing, seven are stable, and nine have insufficient data, as of 2017. ")
        
        example2 = (f"Given claim is: Global warming is driving polar bears toward extinction"
                   f"Sentences supporting the given claim from the given textual document are: Environmental impacts include the extinction or relocation of many species as their ecosystems change, most immediately the environments of coral reefs, mountains, and the Arctic. Rising global temperatures, caused by the greenhouse effect, contribute to habitat destruction, endangering various species, such as the polar bear. ")

        example3 = (f"Given claim is: the bushfires [in Australia] were caused by arsonists and a series of lightning strikes, not 'climate change'."
                    f"Sentences failed to support or refute the given claim from the given textual document are:  The 2007 Kangaroo Island bushfires were a series of bushfires caused by lightning strikes on 6 December 2007 on Kangaroo Island, South Australia, resulting in the destruction of 95,000 hectares (230,000 acres) of national park and wilderness protection area. Many fires are as a result of either deliberate arson or carelessness, however these fires normally happen in readily accessible areas and are rapidly brought under control. Man-made events include arcing from overhead power lines, arson, accidental ignition in the course of agricultural clearing, grinding and welding activities, campfires, cigarettes and dropped matches, sparks from machinery, and controlled burn escapes. The fires would have been caused by both natural phenomenon and human hands. A summer heat wave in Victoria, Australia, created conditions which fuelled the massive bushfires in 2009.")

        # example = example.replace("\n", "")
        prompt = (f"<s> [INST] You are an expert in stance detection. "
                  f"You only have three options (REFUTES, SUPPORTS, and NOT ENOUGH INFO) to detect stance from a textual evidence: refuting, supporting, or not finding enough information for the given claim."
                  f"Check the evidences on sentence level."
                  f"Only output must be one of these three options: (1) REFUTES, (2) SUPPORTS, or (3) NOT ENOUGH INFO. \n"
                  f"Examples are the following:"
                  f"Example 1: {example1}\n[/INST]"
                  f"Answer: REFUTES."    
                  f"[INST]Example 2: {example2}\n[/INST]"
                  f"Answer: SUPPORTS."
                  f"[INST]Example 3: {example3}\n[/INST]"
                  f"Final Answer: NOT ENOUGH INFO. "
                  f"[INST]You should not output more than one word, i.e., (1) REFUTES, (2) SUPPORTS, or (3) NOT ENOUGH INFO. "
                  f"Do not output true or false, rather (1) REFUTES, (2) SUPPORTS, or (3) NOT ENOUGH INFO, for refuting, supporting, or not finding enough information for the given claim.  "
                  f"The only output must be one of these three options: (1) REFUTES, (2) SUPPORTS, or (3) NOT ENOUGH INFO."
                  f"The output should not contain explanations, notes, or numbers, and it should not begin with a number. Only output correct answer, otherwise output NOT ENOUGH INFO.[/INST]</s>\n"
                  f"Do not consider any other knowledge except the provided textual evidence."
                  f"[INST]The given claim is: {claim}\n Given textual evidence is: {text} [/INST] ")
#         example1 = (f"Given claim is: The polar bear population has been growing. "
#                    f"Textual document is: \"Ask the experts: Are polar bear populations increasing?\" The growth of the human population in the Eurasian Arctic in the 16th and 17th century, together with the advent of firearms and increasing trade, dramatically increased the harvest of polar bears. The numbers taken grew rapidly in the 1960s, peaking around 1968 with a global total of 1,250 bears that year. In two areas where harvest levels have been increased based on increased sightings, science-based studies have indicated declining populations, and a third area is considered data-deficient. Of the 19 recognized polar bear subpopulations, one is in decline, two are increasing, seven are stable, and nine have insufficient data, as of 2017. ")
        
#         example2 = (f"Given claim is: Global warming is driving polar bears toward extinction"
#                    f"Textual document is: \"Recent Research Shows Human Activity Driving Earth Towards Global Extinction Event\" Environmental impacts include the extinction or relocation of many species as their ecosystems change, most immediately the environments of coral reefs, mountains, and the Arctic. Rising temperatures push bees to their physiological limits, and could cause the extinction of bee populations. Rising global temperatures, caused by the greenhouse effect, contribute to habitat destruction, endangering various species, such as the polar bear. \"Bear hunting caught in global warming debate\". ")

#         example3 = (f"Given claim is: the bushfires [in Australia] were caused by arsonists and a series of lightning strikes, not 'climate change'."
#                     f"Textual evidence is:  The 2007 Kangaroo Island bushfires were a series of bushfires caused by lightning strikes on 6 December 2007 on Kangaroo Island, South Australia, resulting in the destruction of 95,000 hectares (230,000 acres) of national park and wilderness protection area. Many fires are as a result of either deliberate arson or carelessness, however these fires normally happen in readily accessible areas and are rapidly brought under control. Man-made events include arcing from overhead power lines, arson, accidental ignition in the course of agricultural clearing, grinding and welding activities, campfires, cigarettes and dropped matches, sparks from machinery, and controlled burn escapes. The fires would have been caused by both natural phenomenon and human hands. A summer heat wave in Victoria, Australia, created conditions which fuelled the massive bushfires in 2009.")

#         # example = example.replace("\n", "")
#         prompt = (f"<s> [INST] You are an expert in stance detection. "
#                   f"You only have three options (REFUTES, SUPPORTS, and NOT ENOUGH INFO) to detect stance from a textual evidence: refuting, supporting, or not finding enough information for the given claim."
#                   f"Only output must be one of these three options: (1) REFUTES, (2) SUPPORTS, or (3) NOT ENOUGH INFO. \n"
#                   f"Examples are the following:"
#                   f"Example 1: {example1}\n[/INST]"
#                   f"Answer: REFUTES."    
#                   f"[INST]Example 2: {example2}\n[/INST]"
#                   f"Answer: SUPPORTS."
#                   f"[INST]Example 3: {example3}\n[/INST]"
#                   f"Final Answer: NOT ENOUGH INFO. "
#                   f"[INST]You should not output more than one word, i.e., (1) REFUTES, (2) SUPPORTS, or (3) NOT ENOUGH INFO. "
#                   f"Do not output true or false, rather (1) REFUTES, (2) SUPPORTS, or (3) NOT ENOUGH INFO, for refuting, supporting, or not finding enough information for the given claim.  "
#                   f"The only output must be one of these three options: (1) REFUTES, (2) SUPPORTS, or (3) NOT ENOUGH INFO."
#                   f"The output should not contain explanations, notes, or numbers, and it should not begin with a number. Only output correct answer, otherwise output NOT ENOUGH INFO.[/INST]</s>\n"
#                   f"Do not consider any other knowledge except the provided textual evidence."
#                   f"[INST]The given claim is: {claim}\n Given textual evidence is: {text} [/INST] ")

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

model = 'mix'
data2 = model+'_llm_output.jsonl'
input_file = 'climate-fever.jsonl'
#
path = '/local/upb/users/u/uqudus/profiles/unix/cs/fever/climate-fever/'
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
            ev2 = ''
            for dd in data['evidences']:
                ev2 = ev2 +' '+ dd['evidence']
            
            print(str(claim))
            print(str(ev2))
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

