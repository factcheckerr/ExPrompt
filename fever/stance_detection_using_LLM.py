import json

import requests

from nltk.tokenize import sent_tokenize
def tokenize_into_sentences(paragraph):
    sentences = sent_tokenize(paragraph)
    return sentences






    
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
        #         f"You have only 3 options (REFUTES, SUPPORTS, and NOT ENOUGH INFO) to detect the stance from the following textual description representing the given claim."
        #         f"[/INST] Model answer</s> [INST] "
        #         f"Given claim: {claim}."
        #         f"Textual description: {text}."
        #         f"Answer only a single option from the three: (1) REFUTES, (2) SUPPORTS, and (3) NOT ENOUGH INFO. "
        #         f"Provide no explanations or write no notes.[/INST]")
        
        example = """Given claim is: Brad Wilk was a drummer for Greta. textual description is: Brad Wilk (born September 5, 1968) is an American drummer. He is best known as a member of the rock bands Rage Against the Machine (1991–2000, 2007–2011, 2019–present), Audioslave (2001–2007, 2017), and Prophets of Rage (2016–2019). Wilk started his career as a drummer for Greta in 1990, and helped co-found Rage Against the Machine with Tom Morello and Zack de la Rocha in August 1991. Following that band's breakup in October 2000, Wilk, Morello, Rage Against the Machine bassist Tim Commerford and Soundgarden frontman Chris Cornell formed the supergroup Audioslave, which broke up in 2007. From 2016 to 2019, he played in the band Prophets of Rage, with Commerford, Morello, Chuck D, B-Real and DJ Lord. He has played with Rage Against the Machine since their reunion. Wilk has also performed drums on English metal band Black Sabbath's final album 13, released in June 2013. He briefly played with Pearl Jam shortly after the release of their debut album Ten. Early life
Wilk was born on September 6, 1968, in Portland, Oregon. He was raised in Chicago, Illinois, before his family settled in Southern California. He started to play the drums when he was thirteen years old. He has cited John Bonham, Keith Moon, and Elvin Jones as his greatest influences. Wilk was a fan of Van Halen in his youth. Career
Rage Against the Machine (1991–2000, 2007–2011, 2019–present)
Wilk's success as the drummer of Rage Against the Machine came from the failure of a different band; he once auditioned for a band called Lock Up, who released one album (titled Something Bitchin' This Way Comes) through Geffen records in 1989 and broke up when the album received little media attention upon release. Former Lock Up guitarist Tom Morello was looking to pick up where Lock Up left off and start a new band, and contacted Wilk, who was playing with the band Greta, to see if he was interested in playing the drums. A short while after, the duo met Zack de la Rocha while he was rapping freestyle in a club, and through him, bassist Tim Commerford (a childhood friend of de la Rocha). The band played two shows in 1991, and spent 1992 frequenting the L. A. club circuit, during which they signed a record deal with Epic Records, and released their self-titled debut album that November. They quickly achieved commercial success and would go on to release three more studio albums–Evil Empire in 1996, The Battle of Los Angeles in 1999, and Renegades in 2000– before disbanding in October 2000. Rage Against the Machine reunited to play at the Coachella Music Festival in Coachella, California on January 22, 2007. On April 29, 2007, Rage Against the Machine reunited at the Coachella Music Festival (Rage Against the Machine reunion tour). 
Finally the answer is: SUPPORTS
"""
        
        prompt=(f"<s> [INST] You are an expert in stance detection. "
                f"You have only 3 options (REFUTES, SUPPORTS, and NOT ENOUGH INFO) to detect stance from a textual description representing the given claim. "
                f"Please answer only a single option from the three: (1) REFUTES, (2) SUPPORTS, and (3) NOT ENOUGH INFO. "
                f"[/INST] Model answer</s> [INST]\n Given textual description is: {text} \nThe given claim is: {claim}\n "
                f"You should not output more than one word, i.e., (1) REFUTES, (2) SUPPORTS, or (3) NOT ENOUGH INFO. Do not output true or false, rather (1) REFUTES, (2) SUPPORTS, or (3) NOT ENOUGH INFO.  Only output must be from one of these three options: (1) REFUTES, (2) SUPPORTS, or (3) NOT ENOUGH INFO. Provide no explanations or write no notes or output no numbers. Answer should not start with a number. For example: {example} [/INST]\n")
        # print(prompt)
        response = requests.get(url=self.url,
                                headers={"accept": "application/json", "Content-Type": "application/json"},
                                json={"model": self.model, "prompt": prompt})
        # print(response.json()["response"])
        return response.json()["response"]

    

    def get_triples_from_transformer(self, text: str, claim: str):
        """
        :param text: String representation of an OWL Class Expression
        """

        prompt=(f"<s> [INST] You are an expert in linguistics and stance detection. "
                    f"Given claim: {claim}."
                    f"You have only 3 options (REFUTES, SUPPORTS, and NOT ENOUGH INFO) to detect the stance from the following textual description representing the given claim."
                    f"Textual description: {text}."
                    f"[/INST] Model answer</s> [INST] "
                    f"Answer only a single option from the three: (1) REFUTES, (2) SUPPORTS, and (3) NOT ENOUGH INFO. "
                    f"Provide no explanations or write no notes.[/INST]")

        # print(prompt)
        print("--------------------------------------------------------------LLM Starts here!!!--------------------------------------------")
    
        response = pipe(prompt)[0]['generated_text']

        response = response.split("[/INST]")[-1]
        
        # response = response.split(" (")[1]
        return response

    
    




claim_lines = []
# Read data from a file (assuming it's in JSON format)
count =0

with open('llm_output_train.jsonl', 'r') as output_file:
    for line in output_file:
        # print(line)
        cdata = json.loads(line)
        cc = cdata['claim']
        print(cc)
        claim_lines.append(cc)
        
with open('output_fever_train.jsonl', 'r') as f,  open('llm_output_train.jsonl', 'a') as output_file:
    for line in f:
        data = json.loads(line)
        claim = data['claim']
        if claim not in claim_lines:
            claim_lines.append(claim)
            text = data['text']
            print("claim:"+ claim)
            print("ground truth label:"+ data['label'])
            sentences = tokenize_into_sentences(text.replace("\n"," "))
            # print(" ".join(sentences[:5]))
            # exit(1)
            print(len(sentences))
            text = " ".join(sentences[:500])
            
            verbalizer = LLMStanceDetector(text=text, claim= claim)
            answer = verbalizer.get_response_from_api_call()
            
            answer = answer.replace("\n"," ")
            # print("answer is:"+ answer)
            final_verdict = "NOT ENOUGH INFO"
            if "Answer: REFUTES" in answer or ( "no evidence that supports the claim".lower()  in answer.lower() or ("REFUTE".lower() in answer.lower() and "SUPPORT".lower() not in answer.lower())):
                final_verdict = "REFUTES"
                print("ANSWER->REFUTES")
            elif "Answer: SUPPORTS" in answer or ( "SUPPORT".lower() in answer.lower() and "REFUTE".lower() not in answer.lower()):
                final_verdict = "SUPPORTS"
                print("ANSWER->SUPPORTS")
            elif "NOT ENOUGH INFO".lower() in answer.lower():
                print("ANSWER->NOT ENOUGH INFO")
                final_verdict = "NOT ENOUGH INFO"
            else:
                print(text)
                print("answer is: "+answer)
                break

            data['LLM_prediction'] = final_verdict

            # Write the updated JSON object to the output file
            json.dump(data, output_file)
            output_file.write('\n')

            # print(data)
            print("--------------------------------claim ENDS---------------------------")
            count = count +1
            # if count >1000:
            #     break

# print(lines)
        # exit(1)
    
