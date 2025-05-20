from openai import OpenAI
import time


COMMENT_SYSTEM_PROMPT = """
As a Code Commenting Reviewer, your responsibility is to rigorously assess the quality of comments within the code. Your role is strictly evaluative—you do not modify or suggest changes to the code or its comments. The following criteria define high standards for code commenting quality:
- Comments must be clear, concise, and unambiguous, with fluent and professional language. Redundancy, vagueness, or awkward phrasing is not acceptable.
- Technical terminology must be used accurately and appropriately explained where necessary. All terms must align with the underlying logic and functionality of the code.
- Complex algorithms or business logic must be thoroughly documented, including essential background information, to facilitate understanding without requiring additional context.
- Comments must clearly state the function and purpose of code blocks, allowing readers to comprehend the code's intention without having to parse the implementation details.
- Critical logic and algorithmic steps must be explicitly annotated, including conceptual explanations and step-by-step clarifications where applicable.
- Any edge case handling, exception management, or special condition logic must be clearly described to ensure reliability and maintainability.
- All comments must adhere to project or industry documentation standards (e.g., Python Docstring, Javadoc, Doxygen), maintaining consistent format and best practices throughout the codebase.
- Comment density must be proportional to code complexity—sufficiently informative for intricate logic, while avoiding over-commenting in straightforward sections.
- Redundant, outdated, or repetitive comments are not acceptable. Every comment must be purposeful, current, and contribute to the clarity and maintainability of the code.
"""

COMMENT_USER_PROMPT = """
# Code Comment Quality Evaluation Criteria (Total Score: 9 Points)

## 1. Comment Coverage (3 Points)

- **First**, identify all key parts of the code that warrant comments — such as class and function definitions, complex or non-obvious logic, important configurations or constants, and critical control flows or exception handling. Let `a` represent the total number of key positions where comments **should** be provided.  
- Next, align the previously identified key parts of the code one by one, and count how many of these key positions are actually accompanied by meaningful comments. Let b denote the number of key positions that do have comments, rather than the total number of all comments.
- Compute the coverage ratio as `b / a`.  
- Final score is calculated as: `score1 = 3 × b / a`.  (Capped at 3 points)  

## 2. Clarity of Expression (3 Points)  
Assess whether the existing comments accurately, thoroughly, and intuitively convey the intention and logic of the code, while avoiding vagueness, redundancy, ambiguity, or misleading expressions.  
Only the existing comments are evaluated (uncommented parts are not considered). The score is scaled based on comment quality and density.  
- Each comment starts with a base score of `1 point`.  
- Deduct `0.25 points` for each minor ambiguity.  
- Deduct `0.5 point` for each severe lack of clarity.  
- Let `x` be the sum of all raw scores, and `b` the total number of comments.  
- Calculate the average raw score: `p = x / b`.  
- Final score is: `score2 = 3 × p × (b / a)`. (Capped at 3 points)  
- If `a = 0`, the default score is a full 3 points.

## 3. Format Compliance (3 Points)  
Evaluate whether the comments follow the conventions of the target programming language or project style guide, including format, indentation, consistency, placement, and overall style.  
Only the existing comments are evaluated (uncommented parts are not considered). The score is scaled based on formatting quality and comment density.  
- Each comment starts with a base score of `1 point`.  
- Deduct `0.25 points` for each minor formatting issue.  
- Deduct `0.5 point` for each severe formatting issue.  
- Let `y` be the sum of all raw scores, and `b` the total number of comments.  
- Calculate the average raw score: `q = y / b`.  
- Final score is: `score3 = 3 × q × (b / a)`. (Capped at 3 points)  
- If the comment style of functions or classes does not follow the specified conventions (e.g., PEP 257, Javadoc, Doxygen), then apply a 1.5-point penalty: `score = 3 × q × (b / a) - 1.5`.  
- If `a = 0`, the default score is a full 3 points.

# Task
Your task is to evaluate the code comment quality of the following solution based on the criteria above.

## solution  
{solution}

## Output format 
Briefly explain the reasoning, then strictly output in the following JSON format.
```
{{
"solution": {{
    "score1": 
    "score2": 
    "score3": 
    "final_score": 
  }},
}}
```

# Output

"""


class OpenAIClient:
    def __init__(self, api_key, base_url, model, system_prompt=None, temperature=0.8):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_answer(self, user_prompt, max_retries=2):
        attempt = 0
        while attempt < max_retries:
            try:
                attempt += 1
             
                # 打印调试信息
                print("=" * 50)
                print(f"🔹 Attempt: {attempt}")
                print(f"🔹 Model: {self.model}")
                print(f"🔹 Base URL: {self.base_url}")
                # print(f"🔹 API KEY: {self.api_key}")
                # print(f"🔹 System Prompt: {self.system_prompt}")
                # print(f"🔹 User Prompt: {user_prompt}")
                print(f"🔹 Temperature: {self.temperature}")
                print("=" * 50)

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": COMMENT_SYSTEM_PROMPT},
                        {"role": "user", "content": COMMENT_USER_PROMPT}
                    ],
                    stream=False,
                    temperature=self.temperature,
                )
           
                return response.choices[0].message.content
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    print("Retrying...")
                    print(f"🔹 API KEY: {self.api_key}")

                    time.sleep(1)
                else:
                    print("Max retries reached, returning 'failed'.")
                    return "failed"

class SystemPrompts:
    def __init__(self, **agents):
        self.agents = agents

    def get_agent(self, agent_name):
        return self.agents.get(agent_name, None)
    

class UserPrompts:
    def __init__(self, **prompts):
        self.prompts = prompts

    def get_prompt(self, prompt_name):
        return self.prompts.get(prompt_name, None)