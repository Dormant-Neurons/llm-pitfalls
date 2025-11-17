# Extracting targets and terms
import pandas as pd
import openai
from pydantic import BaseModel, Field
from keybert import KeyBERT
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from typing import Optional


class TargetDerogatoryTerms(BaseModel):
    derogatory_terms: list[str] = Field(..., description="List of derogatory terms from a hateful text.")
    targets: list[str] = Field(..., description="List of targets mentioned in hateful text")


class TermsTargetsExtractor:

    def __init__(self):
        self.kw_model = KeyBERT()
        nltk.download('wordnet')
        nltk.download('stopwords')  # Download the stopwords dataset (only need to do this once)

        # Get the list of English stop words
        self.stop_words = stopwords.words('english')

        self.client = openai.OpenAI()

        # Outputs
        self.targets = []
        self.derogatory_terms = []

    def _extract_terms(self, cur_text: str):
        extracted_terms = self.kw_model.extract_keywords(cur_text, keyphrase_ngram_range=(1, 2),
                                                         stop_words=self.stop_words)
        return [x for x, _ in extracted_terms if TermsTargetsExtractor.is_novel_term(x)]

    @staticmethod
    def is_novel_term(term):
        # Check if the term exists in WordNet
        synsets = wn.synsets(term)
        return len(synsets) == 0  # If no synsets are found, the term is considered novel

    def _check_hate_with_llm(self, hateful_text, extracted_words):
        try:
            prompt = """
            You are provided with a hateful text and a list of extracted words from this text that could be a target in this hateful text or that could be a derogatory term.

            Your task is to decide which word is a target or derogatory term. Only return the targets or derogatory terms. 

            Text: {text} 

            Extracted words: {terms} 
            """
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4.1-mini-2025-04-14",  # We used: gpt-4.1-mini-2025-04-14
                messages=[  # noqa
                    {
                        "role": "user",
                        "content": prompt.format(text=hateful_text, terms=extracted_words)
                    }
                ],
                response_format=TargetDerogatoryTerms,
            )
            return completion.choices[0].message.parsed

        except Exception as e:
            raise e

    def _main_extract(self, cur_text: str) -> Optional[TargetDerogatoryTerms]:
        extracted_terms_targets = self._extract_terms(cur_text)

        if len(extracted_terms_targets) == 0:
            return None

        out_ = self._check_hate_with_llm(hateful_text=cur_text, extracted_words=extracted_terms_targets)
        print(f"Extracted: {extracted_terms_targets}, LLM says target: {out_.targets}, "
              f"derogatory terms: {out_.derogatory_terms}")

        return out_

    def extract_all_from_dataframe(self, df: pd.DataFrame) -> None:

        df = df[(df['ground_truth'] == 1)].reset_index()

        for i in range(df.shape[0]):
            cur_text = df["text"].iloc[i]
            out_ = self._main_extract(cur_text=cur_text)
            if out_ is not None:
                self.targets.extend(out_.targets)
                self.derogatory_terms.extend(out_.derogatory_terms)

        self.targets = sorted(list(set(self.targets)))
        self.derogatory_terms = sorted(list(set(self.derogatory_terms)))
        print(f"Potentially targets and derogatory terms: {self.targets} + / + {self.derogatory_terms}")

    def extract_from_text(self, cur_text: str):
        out_ = self._main_extract(cur_text=cur_text)
        if out_ is not None:
            self.targets.extend(out_.targets)
            self.derogatory_terms.extend(out_.derogatory_terms)

        self.targets = sorted(list(set(self.targets)))
        self.derogatory_terms = sorted(list(set(self.derogatory_terms)))

