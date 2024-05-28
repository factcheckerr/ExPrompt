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
        #         f"Answer only a single option from the three: (1) CONTRADICTS, (2) SUPPORT, and (3) NOT ENOUGH INFO. "
        #         f"Provide no explanations or write no notes.[/INST]")


        example1 = (f"Given claim is: A diminished ovarian reserve is a very strong indicator of infertility, even in an a priori non-infertile population. "
                   f"Textual document is:  title: Association Between Biomarkers of Ovarian Reserve and Infertility Among Older Women of Reproductive Age "
                   f"content: Importance Despite lack of evidence of their utility, biomarkers of ovarian reserve are being promoted as potential markers of reproductive potential. Objective To determine the associations between biomarkers of ovarian reserve and reproductive potential among women of late reproductive age. Design, Setting, and Participants Prospective time-to-pregnancy cohort study (2008 to date of last follow-up in March 2016) of women (N = 981) aged 30 to 44 years without a history of infertility who had been trying to conceive for 3 months or less, recruited from the community in the Raleigh-Durham, North Carolina, area. Exposures Early-follicular-phase serum level of antim\u00fcllerian hormone (AMH), follicle-stimulating hormone (FSH), and inhibin B and urinary level of FSH. Main Outcomes and Measures The primary outcomes were the cumulative probability of conception by 6 and 12 cycles of attempt and relative fecundability (probability of conception in a given menstrual cycle). Conception was defined as a positive pregnancy test result. Results A total of 750 women (mean age, 33.3 [SD, 3.2] years; 77% white; 36% overweight or obese) provided a blood and urine sample and were included in the analysis. After adjusting for age, body mass index, race, current smoking status, and recent hormonal contraceptive use, women with low AMH values (<0.7 ng/mL [n = 84]) did not have a significantly different predicted probability of conceiving by 6 cycles of attempt (65%; 95% CI, 50%-75%) compared with women (n = 579) with normal values (62%; 95% CI, 57%-66%) or by 12 cycles of attempt (84% [95% CI, 70%-91%] vs 75% [95% CI, 70%-79%], respectively). Women with high serum FSH values (>10 mIU/mL [n = 83]) did not have a significantly different predicted probability of conceiving after 6 cycles of attempt (63%; 95% CI, 50%-73%) compared with women (n = 654) with normal values (62%; 95% CI, 57%-66%) or after 12 cycles of attempt (82% [95% CI, 70%-89%] vs 75% [95% CI, 70%-78%], respectively). Women with high urinary FSH values (>11.5 mIU/mg creatinine [n = 69]) did not have a significantly different predicted probability of conceiving after 6 cycles of attempt (61%; 95% CI, 46%-74%) compared with women (n = 660) with normal values (62%; 95% CI, 58%-66%) or after 12 cycles of attempt (70% [95% CI, 54%-80%] vs 76% [95% CI, 72%-80%], respectively). Inhibin B levels (n = 737) were not associated with the probability of conceiving in a given cycle (hazard ratio per 1-pg/mL increase, 0.999; 95% CI, 0.997-1.001). Conclusions and Relevance Among women aged 30 to 44 years without a history of infertility who had been trying to conceive for 3 months or less, biomarkers indicating diminished ovarian reserve compared with normal ovarian reserve were not associated with reduced fertility. These findings do not support the use of urinary or blood follicle-stimulating hormone tests or antim\u00fcllerian hormone levels to assess natural fertility for women with these characteristics. ")
        example2 = (f"Given claim is: ART substantially reduces infectiveness of HIV-positive people. "
                   f"Textual document is:  title: HIV Treatment as Prevention: Systematic Comparison of Mathematical Models of the Potential Impact of Antiretroviral Therapy on HIV Incidence in South Africa "
                   f"content: BACKGROUND Many mathematical models have investigated the impact of expanding access to antiretroviral therapy (ART) on new HIV infections. Comparing results and conclusions across models is challenging because models have addressed slightly different questions and have reported different outcome metrics. This study compares the predictions of several mathematical models simulating the same ART intervention programmes to determine the extent to which models agree about the epidemiological impact of expanded ART.   \n METHODS AND FINDINGS Twelve independent mathematical models evaluated a set of standardised ART intervention scenarios in South Africa and reported a common set of outputs. Intervention scenarios systematically varied the CD4 count threshold for treatment eligibility, access to treatment, and programme retention. For a scenario in which 80% of HIV-infected individuals start treatment on average 1 y after their CD4 count drops below 350 cells/\u00b5l and 85% remain on treatment after 3 y, the models projected that HIV incidence would be 35% to 54% lower 8 y after the introduction of ART, compared to a counterfactual scenario in which there is no ART. More variation existed in the estimated long-term (38 y) reductions in incidence. The impact of optimistic interventions including immediate ART initiation varied widely across models, maintaining substantial uncertainty about the theoretical prospect for elimination of HIV from the population using ART alone over the next four decades. The number of person-years of ART per infection averted over 8 y ranged between 5.8 and 18.7. Considering the actual scale-up of ART in South Africa, seven models estimated that current HIV incidence is 17% to 32% lower than it would have been in the absence of ART. Differences between model assumptions about CD4 decline and HIV transmissibility over the course of infection explained only a modest amount of the variation in model results.   \n CONCLUSIONS Mathematical models evaluating the impact of ART vary substantially in structure, complexity, and parameter choices, but all suggest that ART, at high levels of access and with high adherence, has the potential to substantially reduce new HIV infections. There was broad agreement regarding the short-term epidemiologic impact of ambitious treatment scale-up, but more variation in longer term projections and in the efficiency with which treatment can reduce new infections. Differences between model predictions could not be explained by differences in model structure or parameterization that were hypothesized to affect intervention impact. ")
        
        example3 = (f"Given claim is: 61% of colorectal cancer patients are diagnosed with regional or distant metastases. "
                   f"Textual document is:  title: Relation between Medicare screening reimbursement and stage at diagnosis for older patients with colon cancer. "
                   f"content: CONTEXT Medicare's reimbursement policy was changed in 1998 to provide coverage for screening colonoscopies for patients with increased colon cancer risk, and expanded further in 2001 to cover screening colonoscopies for all individuals. OBJECTIVE To determine whether the Medicare reimbursement policy changes were associated with an increase in either colonoscopy use or early stage colon cancer diagnosis. DESIGN, SETTING, AND PARTICIPANTS Patients in the Surveillance, Epidemiology, and End Results Medicare linked database who were 67 years of age and older and had a primary diagnosis of colon cancer during 1992-2002, as well as a group of Medicare beneficiaries who resided in Surveillance, Epidemiology, and End Results areas but who were not diagnosed with cancer. MAIN OUTCOME MEASURES Trends in colonoscopy and sigmoidoscopy use among Medicare beneficiaries without cancer were assessed using multivariate Poisson regression. Among the patients with cancer, stage was classified as early (stage I) vs all other (stages II-IV). Time was categorized as period 1 (no screening coverage, 1992-1997), period 2 (limited coverage, January 1998-June 2001), and period 3 (universal coverage, July 2001-December 2002). A multivariate logistic regression (outcome = early stage) was used to assess temporal trends in stage at diagnosis; an interaction term between tumor site and time was included. RESULTS Colonoscopy use increased from an average rate of 285/100,000 per quarter in period 1 to 889 and 1919/100,000 per quarter in periods 2 (P<.001) and 3 (P vs 2<.001), respectively. During the study period, 44,924 eligible patients were diagnosed with colorectal cancer. The proportion of patients diagnosed at an early stage increased from 22.5% in period 1 to 25.5% in period 2 and 26.3% in period 3 (P<.001 for each pairwise comparison). The changes in Medicare coverage were strongly associated with early stage at diagnosis for patients with proximal colon lesions (adjusted relative risk period 2 vs 1, 1.19; 95% confidence interval, 1.13-1.26; adjusted relative risk period 3 vs 2, 1.10; 95% confidence interval, 1.02-1.17) but weakly associated, if at all, for patients with distal colon lesions (adjusted relative risk period 2 vs 1, 1.07; 95% confidence interval, 1.01-1.13; adjusted relative risk period 3 vs 2, 0.97; 95% confidence interval, 0.90-1.05). CONCLUSIONS Expansion of Medicare reimbursement to cover colon cancer screening was associated with an increased use of colonoscopy for Medicare beneficiaries, and for those who were diagnosed with colon cancer, an increased probability of being diagnosed at an early stage. The selective effect of the coverage change on proximal colon lesions suggests that increased use of whole-colon screening modalities such as colonoscopy may have played a pivotal role. ")

        # example = example.replace("\n", "")
        prompt = (f"<s> [INST] You are an expert in stance detection. "
                  f"You only have three options (CONTRADICT, SUPPORT, and NOT ENOUGH INFO) to detect stance from a textual document: contradicting, supporting, or not finding enough information for the given claim. Check contradicting and supporting stances in each sentence and take the maximum."
                  f"Only output must be one of these three options: (1) CONTRADICT, (2) SUPPORT, or (3) NOT ENOUGH INFO. \n"
                  f"Examples are the following:"
                  f"Example 1: {example1}\n[/INST]"
                  f"Answer: CONTRADICT."    
                  f"[INST]Example 2: {example2}\n[/INST]"
                  f"Answer: SUPPORT."
                  f"[INST]Example 3: {example3}\n[/INST]"
                  f"Answer: NOT ENOUGH INFO. "
                  f"[INST]You should not output more than one option, i.e., (1) CONTRADICT, (2) SUPPORT, or (3) NOT ENOUGH INFO. "
                  f"Do not output true or false, rather (1) CONTRADICT, (2) SUPPORT, or (3) NOT ENOUGH INFO, for contradicting, supporting, or not finding enough information for the given claim. "
                  f"The only output must be one of these three options: (1) CONTRADICT, (2) SUPPORT, or (3) NOT ENOUGH INFO."
                  f"The output should not contain explanations, notes, or numbers, and it should not begin with a number. Only output correct answer, otherwise output NOT ENOUGH INFO.[/INST]</s>\n"
                  f"[INST]The given claim is: {claim}\n Given textual document is: {text} [/INST] ")

        # prompt = prompt.replace("\n","")

        # print(prompt)
        response = requests.get(url=self.url,
                                headers={"accept": "application/json", "Content-Type": "application/json"},
                                json={"model": self.model, "prompt": prompt})
        # print(response.json()["response"])
        return response.json()["response"]

    def get_line_from_file(self, file_path, search_pattern):
        # Define the grep command to search for the pattern in the file
        grep_command = ["grep", "-m", "1", str(search_pattern), str(file_path)]

        # Execute the grep command
        result = subprocess.run(grep_command, capture_output=True, text=True)

        # Check if the command was successful
        if result.returncode == 0:
            # Extract and return the output line
            output_lines = result.stdout.splitlines()
            if output_lines:
                return output_lines[0]
            else:
                return None
        else:
            # Print an error message if the command failed
            print("Error executing grep command:", result.stderr)
            return None

    # def get_triples_from_transformer(self, text: str, claim: str):
    #     """
    #     :param text: String representation of an OWL Class Expression
    #     """
    #
    #     prompt=(f"<s> [INST] You are an expert in linguistics and stance detection. "
    #                 f"Given claim: {claim}."
    #                 f"You have only 3 options (CONTRADICT, SUPPORT, and NOT ENOUGH INFO) to detect the stance from the following textual description representing the given claim."
    #                 f"Textual description: {text}."
    #                 f"[/INST] Model answer</s> [INST] "
    #                 f"Answer only a single option from the three: (1) CONTRADICT, (2) SUPPORT, and (3) NOT ENOUGH INFO. "
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
data2 = 'mix_llm_output_dev.jsonl'
input_file = 'claims_dev.jsonl'
#
path = '/local/upb/users/u/uqudus/profiles/unix/cs/fever/scifact/data/'
file_path = path + 'corpus.jsonl'
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
            ev2 = data['cited_doc_ids'][0]
            print(str(claim))
            claim_lines.append(claim)
            line4 = verbalizer.get_line_from_file(file_path, ev2)
            title = json.loads(line4)['title']
            abstract = ' '.join(json.loads(line4)['abstract'])
            text = 'title: '+title + ' content: ' + abstract

            print("claim:" + claim)
            # print("ground truth label:" + data['label'])
            sentences = tokenize_into_sentences(text.replace("\n", " "))
            # print(" ".join(sentences[:5]))
            # exit(1)
            # print(len(sentences))
            # text = " ".join(sentences[:500])
            answer = verbalizer.get_response_from_api_call(text=text, claim=claim)
            answer = answer.replace("\n", " ")
            # print("answer is:"+ answer)
            final_verdict = "NOT ENOUGH INFO"
            if "Answer: CONTRADICT" in answer or ("no evidence that support the claim".lower() in answer.lower() or (
                    "CONTRADICT".lower() in answer.lower() and "SUPPORT".lower() not in answer.lower())):
                final_verdict = "CONTRADICT"
                print("ANSWER->CONTRADICT")
            elif "Answer: SUPPORT" in answer or (
                    "SUPPORT".lower() in answer.lower() and "CONTRADICT".lower() not in answer.lower()):
                final_verdict = "SUPPORT"
                print("ANSWER->SUPPORT")
            elif "NOT ENOUGH INFO".lower() in answer.lower():
                print("ANSWER->NOT ENOUGH INFO")
                final_verdict = "NOT ENOUGH INFO"
            else:
                print(text)
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

