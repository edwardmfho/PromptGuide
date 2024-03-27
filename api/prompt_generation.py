from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

BASE_PROMPT = """
Answer the user query.\n{format_instructions}\n\n
You are an expert in crafting prompt for large
language models. You are tasked with modifying the initial prompt
so that will meet the following requirements:\n

1. The prompt should have a clear task, which define the inital prompt
end goal.

2. The prompt should provide some context about the task, so that it
becomes clear of what the task is about. For example, if the query is 
about cooking a dish, the prompt should provide some context about who
the dish is for, what are the ingredients, what is the occasion, or any
other relevant information such as allergy or dietary restrictions.

3. The prompt should provide some examples of the expected output, for 
example, if the task is to classify a text, the prompt should provide
example such that the model can understand what is expected from it.

4. The prompt should also provide a persona to the model, so that the
model could narrow down to its expertise. For example, if the task is
relevant to summarizing a scientific paper related to microbiology,
a prompt with the persona of a microbiologist, with a specific knowledge
in that area should be provided to the prompt.

5. The prompt should indicate what are the format required. For example,
is the user expecting a long text, a short text, a list, a table, or a 
JSON structured output. 

6. The prompt should be clear on how the tone of the generated content 
should be, for example, casual, formal or be pessimistic. 

Here is the initial prompt that you need to modify:\n
{initial_prompt}
\n\n
END OF INITIAL PROMPT\n\n

Now, you need to create an example prompt such that it meets the above requirements.
You should also ask the users to clarify some of the thing that they are not clear about, 
and provide some suggestion on how to address these ambiguities only based on the 
six requirements we mentioned above.

"""
from langchain_core.pydantic_v1 import BaseModel, Field

class ImprovedPrompt(BaseModel):
    task: str = Field(description="The prompt should have a clear task, which define the end goal.")
    context: str = Field(description="A detailed context about the task")
    examples: str = Field(description="A full list of examples of the expected output")
    persona: str = Field(description="A full sentence describing the persona to the model")
    format_instructions: str = Field(description="The format required for the output")
    tone: str = Field(description="The tone required of the generated content")
    follow_up_question: str = Field(description="List of follow up questions to suggest to think about and to clarify some of the ambiguities")

    
def refine_prompt(initial_prompt: str) -> str:
    model = ChatOpenAI(model="gpt-3.5-turbo-0125")
    output_parser = PydanticOutputParser(pydantic_object=ImprovedPrompt)
    prompt = PromptTemplate(
        template=BASE_PROMPT,
        input_variables=["initial_prompt"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )

    chain = prompt | model | output_parser
    output = chain.invoke({"initial_prompt": initial_prompt})

    response = f"""Task: {output.task}
    Context: {output.context}
    Examples: {output.examples}
    Persona: {output.persona}
    Format Instructions: {output.format_instructions}
    Tone: {output.tone}
    Follow Up Questions: {output.follow_up_question}
    """

    return response