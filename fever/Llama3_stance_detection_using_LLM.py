import json

import requests

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
#         example1 = (f"Given claim is: A diminished ovarian reserve is a very strong indicator of infertility, even in an a priori non-infertile population. "
#                    f"Textual document is:  title: Association Between Biomarkers of Ovarian Reserve and Infertility Among Older Women of Reproductive Age "
#                    f"content: Importance Despite lack of evidence of their utility, biomarkers of ovarian reserve are being promoted as potential markers of reproductive potential. Objective To determine the associations between biomarkers of ovarian reserve and reproductive potential among women of late reproductive age. Design, Setting, and Participants Prospective time-to-pregnancy cohort study (2008 to date of last follow-up in March 2016) of women (N = 981) aged 30 to 44 years without a history of infertility who had been trying to conceive for 3 months or less, recruited from the community in the Raleigh-Durham, North Carolina, area. Exposures Early-follicular-phase serum level of antim\u00fcllerian hormone (AMH), follicle-stimulating hormone (FSH), and inhibin B and urinary level of FSH. Main Outcomes and Measures The primary outcomes were the cumulative probability of conception by 6 and 12 cycles of attempt and relative fecundability (probability of conception in a given menstrual cycle). Conception was defined as a positive pregnancy test result. Results A total of 750 women (mean age, 33.3 [SD, 3.2] years; 77% white; 36% overweight or obese) provided a blood and urine sample and were included in the analysis. After adjusting for age, body mass index, race, current smoking status, and recent hormonal contraceptive use, women with low AMH values (<0.7 ng/mL [n = 84]) did not have a significantly different predicted probability of conceiving by 6 cycles of attempt (65%; 95% CI, 50%-75%) compared with women (n = 579) with normal values (62%; 95% CI, 57%-66%) or by 12 cycles of attempt (84% [95% CI, 70%-91%] vs 75% [95% CI, 70%-79%], respectively). Women with high serum FSH values (>10 mIU/mL [n = 83]) did not have a significantly different predicted probability of conceiving after 6 cycles of attempt (63%; 95% CI, 50%-73%) compared with women (n = 654) with normal values (62%; 95% CI, 57%-66%) or after 12 cycles of attempt (82% [95% CI, 70%-89%] vs 75% [95% CI, 70%-78%], respectively). Women with high urinary FSH values (>11.5 mIU/mg creatinine [n = 69]) did not have a significantly different predicted probability of conceiving after 6 cycles of attempt (61%; 95% CI, 46%-74%) compared with women (n = 660) with normal values (62%; 95% CI, 58%-66%) or after 12 cycles of attempt (70% [95% CI, 54%-80%] vs 76% [95% CI, 72%-80%], respectively). Inhibin B levels (n = 737) were not associated with the probability of conceiving in a given cycle (hazard ratio per 1-pg/mL increase, 0.999; 95% CI, 0.997-1.001). Conclusions and Relevance Among women aged 30 to 44 years without a history of infertility who had been trying to conceive for 3 months or less, biomarkers indicating diminished ovarian reserve compared with normal ovarian reserve were not associated with reduced fertility. These findings do not support the use of urinary or blood follicle-stimulating hormone tests or antim\u00fcllerian hormone levels to assess natural fertility for women with these characteristics. ")
#         example2 = (f"Given claim is: ART substantially reduces infectiveness of HIV-positive people. "
#                    f"Textual document is:  title: HIV Treatment as Prevention: Systematic Comparison of Mathematical Models of the Potential Impact of Antiretroviral Therapy on HIV Incidence in South Africa "
#                    f"content: BACKGROUND Many mathematical models have investigated the impact of expanding access to antiretroviral therapy (ART) on new HIV infections. Comparing results and conclusions across models is challenging because models have addressed slightly different questions and have reported different outcome metrics. This study compares the predictions of several mathematical models simulating the same ART intervention programmes to determine the extent to which models agree about the epidemiological impact of expanded ART.   \n METHODS AND FINDINGS Twelve independent mathematical models evaluated a set of standardised ART intervention scenarios in South Africa and reported a common set of outputs. Intervention scenarios systematically varied the CD4 count threshold for treatment eligibility, access to treatment, and programme retention. For a scenario in which 80% of HIV-infected individuals start treatment on average 1 y after their CD4 count drops below 350 cells/\u00b5l and 85% remain on treatment after 3 y, the models projected that HIV incidence would be 35% to 54% lower 8 y after the introduction of ART, compared to a counterfactual scenario in which there is no ART. More variation existed in the estimated long-term (38 y) reductions in incidence. The impact of optimistic interventions including immediate ART initiation varied widely across models, maintaining substantial uncertainty about the theoretical prospect for elimination of HIV from the population using ART alone over the next four decades. The number of person-years of ART per infection averted over 8 y ranged between 5.8 and 18.7. Considering the actual scale-up of ART in South Africa, seven models estimated that current HIV incidence is 17% to 32% lower than it would have been in the absence of ART. Differences between model assumptions about CD4 decline and HIV transmissibility over the course of infection explained only a modest amount of the variation in model results.   \n CONCLUSIONS Mathematical models evaluating the impact of ART vary substantially in structure, complexity, and parameter choices, but all suggest that ART, at high levels of access and with high adherence, has the potential to substantially reduce new HIV infections. There was broad agreement regarding the short-term epidemiologic impact of ambitious treatment scale-up, but more variation in longer term projections and in the efficiency with which treatment can reduce new infections. Differences between model predictions could not be explained by differences in model structure or parameterization that were hypothesized to affect intervention impact. ")
        
#         example3 = (f"Given claim is: 61% of colorectal cancer patients are diagnosed with regional or distant metastases. "
#                    f"Textual document is:  title: Relation between Medicare screening reimbursement and stage at diagnosis for older patients with colon cancer. "
#                    f"content: CONTEXT Medicare's reimbursement policy was changed in 1998 to provide coverage for screening colonoscopies for patients with increased colon cancer risk, and expanded further in 2001 to cover screening colonoscopies for all individuals. OBJECTIVE To determine whether the Medicare reimbursement policy changes were associated with an increase in either colonoscopy use or early stage colon cancer diagnosis. DESIGN, SETTING, AND PARTICIPANTS Patients in the Surveillance, Epidemiology, and End Results Medicare linked database who were 67 years of age and older and had a primary diagnosis of colon cancer during 1992-2002, as well as a group of Medicare beneficiaries who resided in Surveillance, Epidemiology, and End Results areas but who were not diagnosed with cancer. MAIN OUTCOME MEASURES Trends in colonoscopy and sigmoidoscopy use among Medicare beneficiaries without cancer were assessed using multivariate Poisson regression. Among the patients with cancer, stage was classified as early (stage I) vs all other (stages II-IV). Time was categorized as period 1 (no screening coverage, 1992-1997), period 2 (limited coverage, January 1998-June 2001), and period 3 (universal coverage, July 2001-December 2002). A multivariate logistic regression (outcome = early stage) was used to assess temporal trends in stage at diagnosis; an interaction term between tumor site and time was included. RESULTS Colonoscopy use increased from an average rate of 285/100,000 per quarter in period 1 to 889 and 1919/100,000 per quarter in periods 2 (P<.001) and 3 (P vs 2<.001), respectively. During the study period, 44,924 eligible patients were diagnosed with colorectal cancer. The proportion of patients diagnosed at an early stage increased from 22.5% in period 1 to 25.5% in period 2 and 26.3% in period 3 (P<.001 for each pairwise comparison). The changes in Medicare coverage were strongly associated with early stage at diagnosis for patients with proximal colon lesions (adjusted relative risk period 2 vs 1, 1.19; 95% confidence interval, 1.13-1.26; adjusted relative risk period 3 vs 2, 1.10; 95% confidence interval, 1.02-1.17) but weakly associated, if at all, for patients with distal colon lesions (adjusted relative risk period 2 vs 1, 1.07; 95% confidence interval, 1.01-1.13; adjusted relative risk period 3 vs 2, 0.97; 95% confidence interval, 0.90-1.05). CONCLUSIONS Expansion of Medicare reimbursement to cover colon cancer screening was associated with an increased use of colonoscopy for Medicare beneficiaries, and for those who were diagnosed with colon cancer, an increased probability of being diagnosed at an early stage. The selective effect of the coverage change on proximal colon lesions suggests that increased use of whole-colon screening modalities such as colonoscopy may have played a pivotal role. ")
#         # example = example.replace("\n", "")
#         prompt = (f"<s> [INST] You are an expert in stance detection. "
#                   f"You only have three options (CONTRADICT, SUPPORT, and NOT ENOUGH INFO) to detect stance from a textual document: contradicting, supporting, or not finding enough information for the given claim."
#                   f"Only output must be one of these three options: (1) CONTRADICT, (2) SUPPORT, or (3) NOT ENOUGH INFO. \n"
#                   f"Examples are the following:"
#                   f"Example 1: {example1}\n[/INST]"
#                   f"Answer: CONTRADICT."    
#                   f"[INST]Example 2: {example2}\n[/INST]"
#                   f"Answer: SUPPORT."
#                   f"[INST]Example 3: {example3}\n[/INST]"
#                   f"Final Answer: NOT ENOUGH INFO. "
#                   f"[INST]You should not output more than one word, i.e., (1) CONTRADICT, (2) SUPPORT, or (3) NOT ENOUGH INFO. "
#                   f"Do not output true or false, rather (1) CONTRADICT, (2) SUPPORT, or (3) NOT ENOUGH INFO, for contradicting, supporting, or not finding enough information for the given claim.  "
#                   f"The only output must be one of these three options: (1) CONTRADICT, (2) SUPPORT, or (3) NOT ENOUGH INFO."
#                   f"The output should not contain explanations, notes, or numbers, and it should not begin with a number. Only output correct answer, otherwise output NOT ENOUGH INFO.[/INST]</s>\n"
#                   f"[INST]The given claim is: {claim}\n Given textual document is: {text} \n"
#                   f"The only output must be one of these three options: (1) CONTRADICT, (2) SUPPORT, or (3) NOT ENOUGH INFO."
#                   f"The output should not contain explanations, notes, or numbers, and it should not begin with a number. Only output correct answer, otherwise output NOT ENOUGH INFO.[/INST]\n")



#         example = """Given claim is: Brad Wilk was a drummer for Greta. textual description is: Brad Wilk (born September 5, 1968) is an American drummer. He is best known as a member of the rock bands Rage Against the Machine (1991–2000, 2007–2011, 2019–present), Audioslave (2001–2007, 2017), and Prophets of Rage (2016–2019). Wilk started his career as a drummer for Greta in 1990, and helped co-found Rage Against the Machine with Tom Morello and Zack de la Rocha in August 1991. Following that band's breakup in October 2000, Wilk, Morello, Rage Against the Machine bassist Tim Commerford and Soundgarden frontman Chris Cornell formed the supergroup Audioslave, which broke up in 2007. From 2016 to 2019, he played in the band Prophets of Rage, with Commerford, Morello, Chuck D, B-Real and DJ Lord. He has played with Rage Against the Machine since their reunion. Wilk has also performed drums on English metal band Black Sabbath's final album 13, released in June 2013. He briefly played with Pearl Jam shortly after the release of their debut album Ten. Early life
# Wilk was born on September 6, 1968, in Portland, Oregon. He was raised in Chicago, Illinois, before his family settled in Southern California. He started to play the drums when he was thirteen years old. He has cited John Bonham, Keith Moon, and Elvin Jones as his greatest influences. Wilk was a fan of Van Halen in his youth. Career
# Rage Against the Machine (1991–2000, 2007–2011, 2019–present)
# Wilk's success as the drummer of Rage Against the Machine came from the failure of a different band; he once auditioned for a band called Lock Up, who released one album (titled Something Bitchin' This Way Comes) through Geffen records in 1989 and broke up when the album received little media attention upon release. Former Lock Up guitarist Tom Morello was looking to pick up where Lock Up left off and start a new band, and contacted Wilk, who was playing with the band Greta, to see if he was interested in playing the drums. A short while after, the duo met Zack de la Rocha while he was rapping freestyle in a club, and through him, bassist Tim Commerford (a childhood friend of de la Rocha). The band played two shows in 1991, and spent 1992 frequenting the L. A. club circuit, during which they signed a record deal with Epic Records, and released their self-titled debut album that November. They quickly achieved commercial success and would go on to release three more studio albums–Evil Empire in 1996, The Battle of Los Angeles in 1999, and Renegades in 2000– before disbanding in October 2000. Rage Against the Machine reunited to play at the Coachella Music Festival in Coachella, California on January 22, 2007. On April 29, 2007, Rage Against the Machine reunited at the Coachella Music Festival (Rage Against the Machine reunion tour). 
# Finally the answer is: SUPPORTS
# """
        
#         prompt=(f"<s> [INST] You are an expert in stance detection. "
#                 f"You have only 3 options (REFUTES, SUPPORTS, and NOT ENOUGH INFO) to detect stance from a textual description representing the given claim. "
#                 f"Please answer only a single option from the three: (1) REFUTES, (2) SUPPORTS, and (3) NOT ENOUGH INFO. "
#                 f"[/INST] Model answer</s> [INST]\n Given textual description is: {text} \nThe given claim is: {claim}\n "
#                 f"You should not output more than one word, i.e., (1) REFUTES, (2) SUPPORTS, or (3) NOT ENOUGH INFO. Do not output true or false, rather (1) REFUTES, (2) SUPPORTS, or (3) NOT ENOUGH INFO.  Only output must be from one of these three options: (1) REFUTES, (2) SUPPORTS, or (3) NOT ENOUGH INFO. Provide no explanations or write no notes or output no numbers. Answer should not start with a number. For example: {example} [/INST]\n")
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

    
    
verbalizer = LLMStanceDetector()


model = 'llama3'
data2 = model+'_llm_output_test.jsonl'
input_file = 'output_fever_test.jsonl'

claim_lines = []
# Read data from a file (assuming it's in JSON format)
count =0

with open(data2, 'r') as output_file:
    for line in output_file:
        # print(line)
        cdata = json.loads(line)
        cc = cdata['claim']
        print(cc)
        claim_lines.append(cc)
        
with open(input_file, 'r') as f,  open(data2, 'a') as output_file:
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
            text = " ".join(sentences[:100])
            answer = verbalizer.get_response_from_api_call(text=text, claim= claim)
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
                final_verdict = "INVALID"
                print("answer is: "+answer)
                # break

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
    
