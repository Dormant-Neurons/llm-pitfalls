import os
import openai
from pydantic import BaseModel, Field


def prompt_completion_wave_hate_original(model, text):
    """
    This is the original prompt provided by the paper in their GitHub repo.
    """
    try:
        client = openai.OpenAI()

        prompt = """
        Consider the following comment, and carefully answer the questions in each step to conclude whether it is hate speech or not: 
        comment:"{text}"

        Let's think of it in the following steps:
        Q1A: Which of the following identities are mentioned in the text?
        'race', 'nationality', 'age', 'political', 'religion', 'disability', '(anti-)masker', '(anti-)vaxxer'
        Q1B: Are there any individuals mentioned explicitly by their name?
        Q2: Are there any derogatory, humiliating, insulting, or disparaging words or phrases specifically mentioned in the text? (Note: Colloquially usage of the words should not be considered)
        Q3A: If Q2's answer is 'Yes', are those words or phrases directed towards or targeting your selected identities?
        Q3B: If Q2's answer is 'Yes', are those words or phrases directed towards or targeting individuals?
        Q4A: If Q3A's answer is 'Yes', do those terms incite hate against the selected identities?
        Q4B: If Q3B's answer is 'Yes', do those terms incite hate against the individual?
        Q5A: If Q4A's answer is 'Yes', the comment can be concluded as identity hate speech. Tell me your final conclusion: 'Identity Hate' or 'Non-hate'.
        Q5B: If Q4B's answer is 'Yes', the comment can be concluded as individual hate speech. Tell me your final conclusion: 'Individual Hate' or 'Non-hate'
        """.format(text=text)

        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],  # noqa
        )
        sentence = completion.choices[0].message.content

    except Exception as e:
        sentence = f"Error: {e[:100]}..."

    return sentence


def prompt_verify_output(model, llm_response) -> bool:
    try:
        client = openai.OpenAI()

        prompt = """
        # Identity

        You are checking another LLM's response to a hate speech detection task.

        # Instructions

        * You are provided with a chain of thought reasoning output from the LLM.
        * Your task is to tell me what the final response is: HATE or NON-HATE. 
        """

        class HateOutput(BaseModel):
            is_hate: bool = Field(..., description="The final response is: HATE or NON-HATE.")

        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[  # noqa
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": llm_response
                }
            ],
            response_format=HateOutput,
        )
        return completion.choices[0].message.parsed.is_hate

    except Exception as e:
        raise e


def prompt_completion_wave_hate_slightly_adjusted(model, text):
    """
    This is a slightly adjusted prompt provided by the paper in their GitHub repo. With *hate* definition from paper.
    """
    try:
        client = openai.OpenAI()

        prompt = """
        Your task is to analyze if a given text is hate speech or not. Use the following definition of hate:
        "Hatred, hostility, or violence towards member(s) of a race, ethnicity, nation, religion, gender, gender identity, sexual orientation or any other designated sector of society"

        Consider the following comment, and carefully answer the questions in each step to conclude whether it is hate speech or not: 
        comment:"{text}"

        Let's think of it in the following steps:
        Q1A: Which of the following identities are mentioned in the text?
        'race', 'nationality', 'age', 'political', 'religion', 'disability', '(anti-)masker', '(anti-)vaxxer'
        Q1B: Are there any individuals mentioned explicitly by their name?
        Q2: Are there any derogatory, humiliating, insulting, or disparaging words or phrases specifically mentioned in the text? (Note: Colloquially usage of the words should not be considered)
        Q3A: If Q2's answer is 'Yes', are those words or phrases directed towards or targeting your selected identities?
        Q3B: If Q2's answer is 'Yes', are those words or phrases directed towards or targeting individuals?
        Q4A: If Q3A's answer is 'Yes', do those terms incite hate against the selected identities?
        Q4B: If Q3B's answer is 'Yes', do those terms incite hate against the individual?
        Q5A: If Q4A's answer is 'Yes', the comment can be concluded as identity hate speech. Tell me your final conclusion: 'Identity Hate' or 'Non-hate'.
        Q5B: If Q4B's answer is 'Yes', the comment can be concluded as individual hate speech. Tell me your final conclusion: 'Individual Hate' or 'Non-hate'
        """.format(text=text)

        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],  # noqa
        )
        sentence = completion.choices[0].message.content

    except Exception as e:
        sentence = f"Error: {e[:100]}..."

    return sentence


def prompt_completion_wave_hate_original_with_terms_targets(model, text, target_terms, derogatory_terms):
    """
    This is the original prompt provided by the paper in their GitHub repo + adjustment to include the terms and targets.
    """
    try:
        client = openai.OpenAI()

        prompt = """
        Consider the following comment, and carefully answer the questions in each step to conclude whether it is hate speech or not: 
        comment:"{text}"

        Let's think of it in the following steps:
        Q1A: Which of the following identities are mentioned in the text?
        'race', 'nationality', 'age', 'political', 'religion', 'disability', '(anti-)masker', '(anti-)vaxxer'
        Q1B: Are there any individuals mentioned explicitly by their name?
        Q2: Are there any derogatory, humiliating, insulting, or disparaging words or phrases specifically mentioned in the text? (Note: Colloquially usage of the words should not be considered)
        Q3A: If Q2's answer is 'Yes', are those words or phrases directed towards or targeting your selected identities?
        Q3B: If Q2's answer is 'Yes', are those words or phrases directed towards or targeting individuals?
        Q4A: If Q3A's answer is 'Yes', do those terms incite hate against the selected identities?
        Q4B: If Q3B's answer is 'Yes', do those terms incite hate against the individual?
        Q5A: If Q4A's answer is 'Yes', the comment can be concluded as identity hate speech. Tell me your final conclusion: 'Identity Hate' or 'Non-hate'.
        Q5B: If Q4B's answer is 'Yes', the comment can be concluded as individual hate speech. Tell me your final conclusion: 'Individual Hate' or 'Non-hate'

        Here is a list of targets which were used in previous hate speech. They might help you to decide if the comment is hate speech or not:
        {list_of_targets}.

        Here is a list of derogatory terms which were used in previous hate speech. They might help you to decide if the comment is hate speech or not:
        {list_of_hateful_words}.
        """.format(text=text, list_of_targets=", ".join(target_terms),
                   list_of_hateful_words=", ".join(derogatory_terms))

        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],  # noqa
        )
        sentence = completion.choices[0].message.content

    except Exception as e:
        sentence = f"Error: {e[:100]}..."

    return sentence



